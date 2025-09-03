# robust_mom_bootstrap.py
import os
import math
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Iterable
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from data.DatasetLoader import DatasetLoader

# ============================================================
#                    REPRO & GLOBALS
# ============================================================
RNG_SEED = 123
np.random.seed(RNG_SEED)
random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)

ANNUALIZATION = 252
TARGET_VOL = 0.10
EWMA_SPAN = 90

# ============================================================
#                       BOOTSTRAP
# ============================================================
def create_circular_blocks(ts_tensor: torch.Tensor, block_size: int):
    """Create list of overlapping circular blocks of length block_size."""
    T = ts_tensor.shape[0]
    ext = torch.vstack([ts_tensor, ts_tensor[:block_size, :]])
    return [ext[i:i + block_size, :] for i in range(T)]

def sample_from_blocks(blocks, N, block_size, rng=None):
    """Sample enough blocks to cover N rows and truncate."""
    b = int(math.ceil(N / block_size))
    choices = (rng.choices if rng else random.choices)(blocks, k=b)
    out = torch.vstack(choices)[:N, :]
    return out

def build_bootstrap_paths_df(train_returns: pd.DataFrame, block_size: int, k: int, seed_base: int = 0):
    """Generate k circular-block bootstrap paths once and return list of DataFrames."""
    ts_tensor = torch.tensor(train_returns.values, dtype=torch.float32)
    T, _ = ts_tensor.shape
    blocks = create_circular_blocks(ts_tensor, block_size)
    idx, cols = train_returns.index, train_returns.columns

    paths = []
    for i in range(k):
        random.seed(seed_base + i * 5)
        samp = sample_from_blocks(blocks, T, block_size)
        paths.append(pd.DataFrame(samp.numpy(), index=idx, columns=cols))
    return paths

# ============================================================
#                       SIGNALS
# ============================================================
def tsmom_signal_from_returns(returns: pd.DataFrame, lookback: int, method: str = "prod"):
    """
    Time Series Momentum (Moskowitz–Ooi–Pedersen):
    sign of past cumulative return over lookback (lagged by 1 for execution).
    method='prod': (1+r).rolling(L).apply(np.prod, raw=True) - 1.0
    method='sum' : simple rolling sum of returns (approx)
    Returns positions in {-1,0,1}
    """
    if method == "prod":
        mom = (1.0 + returns).rolling(lookback).apply(np.prod, raw=True) - 1.0
    else:
        mom = returns.rolling(lookback).sum()
    mom = mom.shift(1)  # use info known at t-1
    pos = np.sign(mom).replace({-0.0: 0.0})
    return pos

# Top-level wrapper (picklable) to avoid lambda in __main__
def signal_fn_prod(rets: pd.DataFrame, L: int) -> pd.DataFrame:
    return tsmom_signal_from_returns(rets, L, method="prod")

# ============================================================
#                       BACKTEST + METRICS
# ============================================================
def backtest_positions(returns: pd.DataFrame, positions: pd.DataFrame,
                       vol_target_annual=TARGET_VOL, vol_span=EWMA_SPAN):
    """Next-day execution; returns DataFrame with 'portfolio' and 'portfolio_scaled'."""
    pnl = returns.shift(-1) * positions
    port = pnl.mean(axis=1)  # equal-weight across instruments
    ann_vol = port.ewm(span=vol_span, min_periods=vol_span).std() * np.sqrt(ANNUALIZATION)
    ann_vol = ann_vol.shift(1).replace(0, np.nan)
    scaled = port * (vol_target_annual / (ann_vol + 1e-12))
    out = pd.DataFrame({"portfolio": port, "portfolio_scaled": scaled}).dropna()
    return out

def sharpe(series: pd.Series):
    x = series.dropna()
    if x.std() == 0:
        return np.nan
    return float(x.mean() / x.std() * np.sqrt(ANNUALIZATION))

def sortino(series: pd.Series):
    x = series.dropna()
    downside = x[x < 0].std()
    if downside == 0 or np.isnan(downside):
        return np.nan
    return float(x.mean() / downside * np.sqrt(ANNUALIZATION))

def max_drawdown(series: pd.Series):
    x = (1 + series.fillna(0)).cumprod()
    peak = x.cummax()
    dd = x / peak - 1.0
    return float(dd.min())

def avg_drawdown(series: pd.Series):
    x = (1 + series.fillna(0)).cumprod()
    peak = x.cummax()
    dd = x / peak - 1.0
    return float(dd.mean())

def metrics(series: pd.Series):
    return {
        "E[R]": series.mean() * ANNUALIZATION,
        "Std(R)": series.std() * np.sqrt(ANNUALIZATION),
        "Sharpe": sharpe(series),
        "Sortino": sortino(series),
        "MaxDD": max_drawdown(series) * 100.0,
        "AvgDD": avg_drawdown(series) * 100.0,
    }

# -------- Example utility functions (pass any into select_param_empirical_parallel) --------
def util_sharpe_scaled(bt: pd.DataFrame) -> float:
    return sharpe(bt["portfolio_scaled"])

def util_sharpe_unscaled(bt: pd.DataFrame) -> float:
    return sharpe(bt["portfolio"])

def util_sortino_scaled(bt: pd.DataFrame) -> float:
    return sortino(bt["portfolio_scaled"])

def util_neg_maxdd_scaled(bt: pd.DataFrame) -> float:
    return -max_drawdown(bt["portfolio_scaled"])

def util_combo(bt: pd.DataFrame, alpha=1.0, beta=0.0) -> float:
    s = sharpe(bt["portfolio_scaled"])
    mdd = max_drawdown(bt["portfolio_scaled"])
    return alpha * s - beta * abs(mdd)

# ============================================================
#                PARAMETER SELECTION ROUTINES
# ============================================================
def _eval_param_bootstrap_single(param: int,
                                 paths: List[pd.DataFrame],
                                 signal_fn: Callable[[pd.DataFrame, int], pd.DataFrame]) -> Tuple[int, float, float]:
    """Helper for parallel bootstrap table: compute mean Sharpe/MaxDD over paths for one param."""
    sh_list, dd_list = [], []
    for boot_ret in paths:
        pos = signal_fn(boot_ret, param)
        bt = backtest_positions(boot_ret, pos)
        sh_list.append(sharpe(bt["portfolio_scaled"]))
        dd_list.append(max_drawdown(bt["portfolio_scaled"]))
    return int(param), float(np.nanmean(sh_list)), float(np.nanmean(dd_list))

def select_params_via_bootstrap(paths: List[pd.DataFrame], param_trials, signal_fn) -> pd.DataFrame:
    """Sequential version (kept for reference)."""
    rows = []
    for param in tqdm(param_trials, desc="Evaluating parameters"):
        _, sh, dd = _eval_param_bootstrap_single(param, paths, signal_fn)
        rows.append({"param": int(param), "sharpe": sh, "maxDD": dd})
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)

def select_params_via_bootstrap_parallel(paths: List[pd.DataFrame],
                                         param_trials,
                                         signal_fn,
                                         n_jobs: int = None,
                                         backend: str = "process") -> pd.DataFrame:
    """Parallel bootstrap table over parameters."""
    Executor = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
    rows = []
    with Executor(max_workers=n_jobs) as ex:
        futs = [ex.submit(_eval_param_bootstrap_single, int(p), paths, signal_fn) for p in param_trials]
        for fut in as_completed(futs):
            param, sh, dd = fut.result()
            rows.append({"param": param, "sharpe": sh, "maxDD": dd})
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)

def _eval_param_single(param: int,
                       train_returns: pd.DataFrame,
                       signal_fn: Callable[[pd.DataFrame, int], pd.DataFrame],
                       utility_fn: Callable[[pd.DataFrame], float]) -> Tuple[int, float]:
    """Evaluate one parameter: build positions -> backtest -> utility."""
    pos = signal_fn(train_returns, param)
    bt  = backtest_positions(train_returns, pos)
    util = utility_fn(bt)
    if util is None or not np.isfinite(util):
        util = -np.inf
    return int(param), float(util)

def select_param_empirical_parallel(train_returns: pd.DataFrame,
                                    param_trials: Iterable[int],
                                    signal_fn: Callable[[pd.DataFrame, int], pd.DataFrame],
                                    utility_fn: Callable[[pd.DataFrame], float],
                                    n_jobs: int = None,
                                    backend: str = "process") -> Tuple[int, float]:
    """
    Parallel ERM selection (maximizes a user-provided utility on TRAIN).
    """
    Executor = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
    best_param, best_util = None, -np.inf
    with Executor(max_workers=n_jobs) as ex:
        futs = [ex.submit(_eval_param_single, int(p), train_returns, signal_fn, utility_fn) for p in param_trials]
        for fut in as_completed(futs):
            param, util = fut.result()
            if util > best_util:
                best_util = util
                best_param = param
    return best_param, float(best_util)

def pick_percentile_param(df: pd.DataFrame, q):
    """
    q can be 'max' or a percentile in (0,1].
    'max' = best Sharpe. Otherwise, pick parameter at the given upper-tail percentile.
    """
    if q == "max":
        return int(df.iloc[0]['param'])
    idx = int(np.clip(np.ceil((1.0 - q) * (len(df) - 1)), 0, len(df) - 1))
    return int(df.iloc[idx]['param'])

def apply_param_to_test(full_returns: pd.DataFrame,
                        train_len: int,
                        lookback: int,
                        signal_fn):
    """Build positions on full series (train+test) and return test-only PnL."""
    positions = signal_fn(full_returns, lookback)
    bt = backtest_positions(full_returns, positions)
    return bt.iloc[train_len:]

# ============================================================
#                        PLOTS
# ============================================================
def plot_gap(df, metric_train, metric_test, gap_col, title, out_path):
    x = df["name"].values
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)

    ax[0].plot(x, df[metric_train], marker="o", label="Train")
    ax[0].plot(x, df[metric_test], marker="o", label="Test")
    ax[0].set_ylabel(title)
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(x, df[gap_col], marker="o", label="Generalization Gap")
    ax[1].set_xlabel("Selection")
    ax[1].set_ylabel("Gap")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    plt.xticks(rotation=30)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

# ============================================================
#                           RUNNER
# ============================================================
if __name__ == "__main__":
    # ------------------ Paths & IO ------------------
    SIGNAL_NAME = "tsmom_moskowitz_prod"
    dataset_name = 'futures'
    continuous_future_method = 'RAD'
    BASE_DIR = os.path.dirname(__file__)
    inputs_path = os.path.join(BASE_DIR, "data", "inputs")
    outputs_path = os.path.join(BASE_DIR, "data", "outputs")
    os.makedirs(os.path.join(outputs_path, "results"), exist_ok=True)

    SIGNAL_NAME = f"{SIGNAL_NAME}_{dataset_name}"

    ds_builder = DatasetLoader(
            flds={
                # commodities
                # 'ZH': ['close'], HEATING OIL has zero prices until 2020
                # 'NR': ['close'], no data starting from 2021
                'CC': ['close'], 'DA': ['close'], 'GI': ['close'], 'JO': ['close'], 'KC': ['close'], 'KW': ['close'],
                'LB': ['close'], 'SB': ['close'], 'ZC': ['close'], 'ZF': ['close'], 'ZZ': ['close'],
                'ZG': ['close'], 'ZI': ['close'], 'ZK': ['close'], 'ZL': ['close'], 'ZN': ['close'],
                'ZO': ['close'], 'ZP': ['close'], 'ZR': ['close'], 'ZT': ['close'], 'ZU': ['close'], 'ZW': ['close'],
                
                # bonds
                # 'EC': ['close'], no data starting from 2021
                'CB': ['close'], 'DT': ['close'], 'FB': ['close'], 'GS': ['close'], 'TU': ['close'], 
                'TY': ['close'], 'UB': ['close'], 'US': ['close'], 'UZ': ['close'], 
                
                # fx
                'AN': ['close'], 'CN': ['close'], 'BN': ['close'], 'DX': ['close'], 'JN': ['close'], 
                'MP': ['close'], 'SN': ['close'],

                # equities
                # 'SP': ['close'] == 'SC': ['close'] == 'ES': ['close'] == S&P500
                'FN': ['close'], 'NK': ['close'], 'ZA': ['close'], 'CA': ['close'], 'EN': ['close'], 'ER': ['close'], 'ES': ['close'],
                'LX': ['close'], 'MD': ['close'], 'XU': ['close'], 'XX': ['close'], 'YM': ['close'],
        }
    )

    data = ds_builder.load_data(
        dataset_name=dataset_name,
        continuous_future_method=continuous_future_method,
    )
    returns = data.pct_change().dropna()

    # Auto workers: leave one CPU free
    N_JOBS = max(1, os.cpu_count() - 1)

    # Candidate lookbacks
    tot = 252
    PARAM_TRIALS = list(range(5, tot + (tot // 2) + 1, 1))
    K_BOOT = 10000
    BLOCK_SIZE = 10
    PICKS = ["max", 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]

    # Choose in-sample utility for ERM baseline
    UTILITY_FN = util_sharpe_scaled
    # UTILITY_FN = util_sortino_scaled
    # UTILITY_FN = util_neg_maxdd_scaled
    # UTILITY_FN = lambda bt: util_combo(bt, alpha=1.0, beta=2.0)

    # ------------------ Train / Test split ------------------
    split_idx = int(len(returns) * 0.80)
    train_returns = returns.iloc[:split_idx].copy()
    test_returns  = returns.iloc[split_idx:].copy()
    full_returns  = returns.copy()

    # ------------------ Experiment Config -------------------
    signal_fn = signal_fn_prod  # picklable

    # Bootstrap once (built on train only)
    bootstrap_paths = build_bootstrap_paths_df(train_returns, BLOCK_SIZE, K_BOOT, seed_base=RNG_SEED)

    # ------------------ Bootstrap selection table (parallel) ------------------
    boot_df = select_params_via_bootstrap_parallel(
        bootstrap_paths,
        PARAM_TRIALS,
        signal_fn=signal_fn,
        n_jobs=N_JOBS,
        backend="process"
    )

    # Pick parameters by ranked percentiles of bootstrap Sharpe
    selected_params = {}
    for q in PICKS:
        name = str(q) if isinstance(q, str) else f"{int(q*100)}th"
        selected_params[name] = pick_percentile_param(boot_df, q)

    # ------------------ Baseline ERM (parallel) ------------------
    L_emp, _ = select_param_empirical_parallel(
        train_returns,
        param_trials=PARAM_TRIALS,
        signal_fn=signal_fn,
        utility_fn=UTILITY_FN,
        n_jobs=N_JOBS,
        backend="process"
    )
    selected_params["ERM_max"] = L_emp

    # ---- Desired, fixed display/evaluation order ----
    DESIRED_ORDER = [f"{p}th" for p in (10,20,30,40,50,60,70,80,90)] + ["max", "ERM_max"]
    ordered_names = [n for n in DESIRED_ORDER if n in selected_params]

    # ------------------ Evaluate train/test IN THIS ORDER ------------------
    records = []
    cumret_panels = {}
    train_len = len(train_returns)

    for name in ordered_names:
        L = selected_params[name]
        # Train
        pos_tr = signal_fn(train_returns, L)
        bt_tr = backtest_positions(train_returns, pos_tr)
        # Test (build positions on full series, slice to test)
        bt_te = apply_param_to_test(full_returns, train_len, L, signal_fn)

        # Metrics on scaled series
        m_tr = metrics(bt_tr["portfolio_scaled"])
        m_te = metrics(bt_te["portfolio_scaled"])

        records.append({
            "name": name, "param": L,
            "Sharpe_train": m_tr["Sharpe"], "Sharpe_test": m_te["Sharpe"],
            "Gap_Sharpe": m_te["Sharpe"] - m_tr["Sharpe"],
            "Sortino_train": m_tr["Sortino"], "Sortino_test": m_te["Sortino"],
            "Gap_Sortino": m_te["Sortino"] - m_tr["Sortino"],
            "MaxDD_train": m_tr["MaxDD"], "MaxDD_test": m_te["MaxDD"],
            "Gap_MaxDD": m_te["MaxDD"] - m_tr["MaxDD"],
            "AvgDD_train": m_tr["AvgDD"], "AvgDD_test": m_te["AvgDD"],
            "Gap_AvgDD": m_te["AvgDD"] - m_tr["AvgDD"],
        })

        # Continuous full-sample cum-returns (scaled) for plotting
        full_pos = signal_fn(full_returns, L)
        full_bt  = backtest_positions(full_returns, full_pos)
        cum = (1 + full_bt["portfolio_scaled"]).cumprod()
        cumret_panels[f"{name}_{L}"] = cum

    # Build DataFrame and lock the categorical order for plotting/printing
    results_df = pd.DataFrame(records)
    results_df["name"] = pd.Categorical(results_df["name"],
                                        categories=ordered_names,
                                        ordered=True)
    results_df = results_df.sort_values("name").reset_index(drop=True)

    print("\n=== Parameter selections & metrics ===")
    print(results_df.round(3).to_string(index=False))

    # Save CSV
    results_path = os.path.join(outputs_path, "results", f"{SIGNAL_NAME}_bootstrap_selection.csv")
    results_df.to_csv(results_path, index=False)

    # ------------------ Plots ------------------
    def plot_gap_local(df, metric_train, metric_test, gap_col, title, fname):
        x = df["name"].values
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)

        ax[0].plot(x, df[metric_train], marker="o", label="Train")
        ax[0].plot(x, df[metric_test], marker="o", label="Test")
        ax[0].set_ylabel(title)
        ax[0].grid(True, alpha=0.3)
        ax[0].legend()

        ax[1].plot(x, df[gap_col], marker="o", label="Generalization Gap")
        ax[1].set_xlabel("Selection")
        ax[1].set_ylabel("Gap")
        ax[1].grid(True, alpha=0.3)
        ax[1].legend()

        plt.xticks(rotation=30)
        out = os.path.join(outputs_path, "results", fname)
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)

    # Generalization gap: Sharpe / Sortino / MaxDD / AvgDD
    plot_gap_local(results_df, "Sharpe_train",  "Sharpe_test",  "Gap_Sharpe",  "Sharpe Ratio",         f"sharpe-gap-{SIGNAL_NAME}.png")
    plot_gap_local(results_df, "Sortino_train", "Sortino_test", "Gap_Sortino", "Sortino Ratio",        f"sortino-gap-{SIGNAL_NAME}.png")
    plot_gap_local(results_df, "MaxDD_train",   "MaxDD_test",   "Gap_MaxDD",   "Max Drawdown (%)",     f"maxdd-gap-{SIGNAL_NAME}.png")
    plot_gap_local(results_df, "AvgDD_train",   "AvgDD_test",   "Gap_AvgDD",   "Average Drawdown (%)", f"avgdd-gap-{SIGNAL_NAME}.png")

    # ------------------ Cumulative returns: Train vs Test panels (each starts at 1) ------------------
    cum_df = pd.DataFrame(cumret_panels)  # full-sample cumrets (scaled) for each selection
    test_start = test_returns.index[0]
    train_start = train_returns.index[0]

    # 1) TRAIN: rebuild incremental cumulative return from the underlying *daily* scaled returns
    # Convert cum_df back to daily scaled returns (r_t = C_t / C_{t-1} - 1), then recompute
    daily_rets = cum_df.pct_change().fillna(0.0)

    # Train cumulative returns: only use train window and rebase to 1 at train_start
    train_rets = daily_rets.loc[train_start:test_start].dropna()
    train_cum = (1.0 + train_rets).cumprod()
    # Ensure exact 1.0 at the first row
    if not train_cum.empty:
        train_cum = train_cum / train_cum.iloc[0]

    # 2) TEST: only use returns from test_start onward and rebase to 1 at test_start
    test_rets = daily_rets.loc[test_start:].dropna()
    test_cum = (1.0 + test_rets).cumprod()
    if not test_cum.empty:
        test_cum = test_cum / test_cum.iloc[0]

    # Plot
    # ------------------ Distinct colors ------------------
    palette = plt.get_cmap('tab20').colors  # 20 high-contrast colors
    color_map = {}
    for i, colname in enumerate(cum_df.columns):
        color_map[colname] = palette[i % len(palette)]

    # ------------------ Build plots ------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)

    def plot_panel(ax, data, title):
        for col in data.columns:
            lw = 1.5
            zorder = 1
            alpha = 0.9
            # Highlight ERM_max
            if col.startswith("ERM_max"):
                lw = 2.5
                zorder = 3
                alpha = 1.0
            ax.plot(data.index, data[col],
                    label=col,
                    color=color_map[col],
                    linewidth=lw,
                    alpha=alpha,
                    zorder=zorder)
        # ax.set_title(title)
        ax.set_ylabel("Cumulative Return")
        # ax.grid(True, alpha=0.3)

        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', labelsize=10)  # increase font size for x-axis
            # ax.tick_params(axis='y', labelsize=11)  # optional, y-axis too

    # select 90th, 50th, and 10th percentiles for train/test panels + ERM_max
    train_cum = train_cum.loc[:,train_cum.columns.str.contains("90|50|10|ERM_max")]
    test_cum = test_cum.loc[:,test_cum.columns.str.contains("90|50|10|ERM_max")]

    plot_panel(ax1, train_cum, "Train Period Cumulative Returns (rebased to 1)")
    plot_panel(ax2, test_cum, "Test Period Cumulative Returns (rebased to 1)")

    # ------------------ Single legend outside ------------------
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles,
               labels,
               loc='lower center',
               bbox_to_anchor=(0.5, -0.10),  # move legend lower
               ncol=len(labels),             # all entries in one row
               fontsize=13)

    # Save
    out = os.path.join(outputs_path, "results", f"cumret-{SIGNAL_NAME}.png")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"\nSaved: {results_path}")
    print("Saved plots to:", os.path.join(outputs_path, "results"))
