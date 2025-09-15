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
import argparse
from typing import Union

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
def backtest_positions(returns: pd.DataFrame,
                       positions: pd.DataFrame,
                       vol_target_annual=TARGET_VOL,
                       vol_span=EWMA_SPAN,
                       vol_scaled_output=True):
    """Next-day execution; returns DataFrame with 'portfolio' and 'portfolio_scaled'."""
    pnl = returns.shift(-1) * positions
    port = pnl.mean(axis=1)  # equal-weight across instruments
    ann_vol = port.ewm(span=vol_span, min_periods=vol_span).std() * np.sqrt(ANNUALIZATION)
    ann_vol = ann_vol.shift(1).replace(0, np.nan)
    scaled = port * (vol_target_annual / (ann_vol + 1e-12))
    out = pd.DataFrame({"portfolio": port, "portfolio_scaled": scaled}).dropna()
    return out['portfolio_scaled'] if vol_scaled_output else out['portfolio']

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
def util_sharpe(bt: pd.Series) -> float:
    return sharpe(bt)

def util_sortino(bt: pd.Series) -> float:
    return sortino(bt)

def util_neg_maxdd(bt: pd.Series) -> float:
    return max_drawdown(bt)

def map_util_fn_code_to_name(code: str) -> str:
    if code == 'util_neg_maxdd':
        return 'MaxDD'
    elif code == 'util_sortino':
        return 'Sortino'
    elif code == 'util_sharpe':
        return 'Sharpe'
    else:
        raise ValueError(f"Unknown utility function code: {code}")

# ============================================================
#                PARAMETER SELECTION ROUTINES
# ============================================================
def _eval_param_bootstrap_single(param: int,
                                 paths: List[pd.DataFrame],
                                 signal_fn: Callable[[pd.DataFrame, int], pd.DataFrame]) -> Tuple[int, float, float]:
    """Helper for parallel bootstrap table: compute mean Sharpe/MaxDD over paths for one param."""
    sh_list, dd_list, sor_list = [], [], []
    for boot_ret in paths:
        pos = signal_fn(boot_ret, param)
        bt_series = backtest_positions(boot_ret, pos)
        sh_list.append(sharpe(bt_series))
        dd_list.append(max_drawdown(bt_series))
        sor_list.append(sortino(bt_series))

    return int(param), float(np.nanmean(sh_list)), float(np.nanmean(dd_list)), float(np.nanmean(sor_list))

def select_params_via_bootstrap(paths: List[pd.DataFrame], param_trials, signal_fn) -> pd.DataFrame:
    """Sequential version (kept for reference)."""
    rows = []
    for param in tqdm(param_trials, desc="Evaluating parameters"):
        _, sh, dd, sor = _eval_param_bootstrap_single(param, paths, signal_fn)
        rows.append({"param": int(param), "Sharpe": sh, "MaxDD": dd, "Sortino": sor})
    return pd.DataFrame(rows).sort_values("Sharpe", ascending=False).reset_index(drop=True)

def select_params_via_robust_bootstrap_parallel(paths: List[pd.DataFrame],
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
            param, sh, dd, sor = fut.result()
            rows.append({"param": param, "Sharpe": sh, "MaxDD": dd, "Sortino": sor})
    return pd.DataFrame(rows).sort_values("Sharpe", ascending=False).reset_index(drop=True)

def _eval_param_classical_bootstrap_single(boot_ret: pd.DataFrame,
                                           params: int,
                                           signal_fn: Callable[[pd.DataFrame, int], pd.DataFrame],
                                           utility_fn: str) -> Tuple[int, float, float]:
    """Helper for parallel bootstrap table: compute mean Sharpe/MaxDD over paths for one param."""
    sh_list, dd_list, sor_list = [], [], []
    for param in params:
        pos = signal_fn(boot_ret, param)
        bt_series = backtest_positions(boot_ret, pos)
        sh_list.append(sharpe(bt_series))
        dd_list.append(max_drawdown(bt_series))
        sor_list.append(sortino(bt_series))
    output_df = pd.DataFrame({"param": params, "Sharpe": sh_list, "MaxDD": dd_list, "Sortino": sor_list})
    output_df = output_df.sort_values(map_util_fn_code_to_name(utility_fn.__name__), ascending=False).reset_index(drop=True)

    param = output_df.iloc[0]['param']
    sr = output_df.iloc[0]['Sharpe']
    md = output_df.iloc[0]['MaxDD']
    so = output_df.iloc[0]['Sortino']

    return int(param), float(sr), float(md), float(so)

def select_params_via_classical_bootstrap_parallel(paths: List[pd.DataFrame],
                                                   param_trials,
                                                   signal_fn,
                                                   utility_fn,
                                                   n_jobs: int = None,
                                                   backend: str = "process") -> pd.DataFrame:
    """Parallel bootstrap table over parameters."""
    Executor = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
    rows = []
    with Executor(max_workers=n_jobs) as ex:
        futs = [ex.submit(_eval_param_classical_bootstrap_single, path, param_trials, signal_fn, utility_fn) for path in paths]
        boot_idx = 0
        for fut in as_completed(futs):
            param, sh, dd, sor = fut.result()
            rows.append({"boot_idx": boot_idx, "param": param, "Sharpe": sh, "MaxDD": dd, "Sortino": sor})
            boot_idx += 1
    return pd.DataFrame(rows).sort_values("Sharpe", ascending=False).reset_index(drop=True)

def _eval_param_single(param: int,
                       train_returns: pd.DataFrame,
                       signal_fn: Callable[[pd.DataFrame, int], pd.DataFrame],
                       utility_fn: Callable[[pd.DataFrame], float]) -> Tuple[int, float]:
    """Evaluate one parameter: build positions -> backtest -> utility."""
    pos = signal_fn(train_returns, param)
    bt_series = backtest_positions(train_returns, pos)
    util = utility_fn(bt_series)
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

def pick_percentile_param(df: pd.DataFrame, q: Union[str, float], metric: str) -> int:
    """
    q can be 'max' or a percentile in (0,1].
    'max' = best Sharpe. Otherwise, pick parameter at the given upper-tail percentile.
    """

    # # sort descending by metric
    # df = df.sort_values(metric, ascending=False).reset_index(drop=True)

    # pick index by quantile
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

    args = argparse.ArgumentParser()
    args.add_argument('--signal', type=str, default='tsmom_moskowitz_prod', help='Signal name')
    args.add_argument('--dataset', type=str, default='futures', help='Dataset name', choices=['futures', 'etfs'])
    args.add_argument('--utility', type=str, default='Sharpe', help='Utility name: Sharpe, Sortino, MaxDD')
    args.add_argument('--method', type=str, default='RAD', help='Continuous future method')
    args.add_argument('--n_boot_samples', type=int, default=10, help='Number of bootstrap samples')
    args.add_argument('--block_size', type=int, default=10, help='Block size for bootstrap')
    parsed = args.parse_args()

    # ------------------ Paths & IO ------------------
    SIGNAL_NAME = parsed.signal
    dataset_name = parsed.dataset
    utility_name = parsed.utility
    continuous_future_method = parsed.method
    N_JOBS = max(1, os.cpu_count() - 1)

    tot = 252
    PARAM_TRIALS = list(range(5, tot + (tot // 2) + 1, 1))
    K_BOOT = parsed.n_boot_samples
    BLOCK_SIZE = parsed.block_size
    PICKS = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]

    BASE_DIR = os.path.dirname(__file__)
    inputs_path = os.path.join(BASE_DIR, "data", "inputs")
    outputs_path = os.path.join(BASE_DIR, "data", "outputs")
    os.makedirs(os.path.join(outputs_path, "results"), exist_ok=True)

    SIGNAL_NAME = f"{SIGNAL_NAME}_{dataset_name}_{utility_name}"

    if dataset_name == 'futures':
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
    elif dataset_name == 'etfs':
        instruments = [
            "SPY","IWM","EEM","TLT","USO","GLD",
            "XLF","XLB","XLK","XLV","XLI","XLU","XLY",
            "XLP","XLE","AGG","DBC","HYG","LQD","UUP"
        ]

        df = pd.read_csv(os.path.join(inputs_path, "sample", "etfs.csv"), sep=";")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")[instruments].resample("B").ffill().dropna()
        returns = df.pct_change().dropna()

    # Choose in-sample utility for ERM baseline
    if utility_name == 'Sharpe':
        UTILITY_FN = util_sharpe
    elif utility_name == 'Sortino':
        UTILITY_FN = util_sortino
    elif utility_name == 'MaxDD':
        UTILITY_FN = util_neg_maxdd
    else:
        raise ValueError(f"Unknown utility_name: {utility_name}")

    # ------------------ Train / Test split ------------------
    all_results = []
    for colname in tqdm(returns.columns, desc="Processing assets..."):
        individual_returns = returns[[colname]].copy().dropna()

        split_idx = int(len(individual_returns) * 0.80)
        train_returns = individual_returns.iloc[:split_idx].copy()
        test_returns  = individual_returns.iloc[split_idx:].copy()
        full_returns  = individual_returns.copy()

        # ------------------ Experiment Config -------------------
        signal_fn = signal_fn_prod  # picklable

        # Bootstrap once (built on train only)
        bootstrap_paths = build_bootstrap_paths_df(train_returns, BLOCK_SIZE, K_BOOT, seed_base=RNG_SEED)

        # ------------------ Robust Bootstrap selection table (parallel) ------------------
        boot_df = select_params_via_robust_bootstrap_parallel(
            bootstrap_paths,
            PARAM_TRIALS,
            signal_fn=signal_fn,
            n_jobs=N_JOBS,
            backend="process"
        )
        # Pick parameters by ranked percentiles of robust bootstrap utility
        selected_params = {}
        for q in PICKS:
            name = str(q) if isinstance(q, str) else f"{int(q*100)}th"
            selected_params[name] = pick_percentile_param(boot_df, q, utility_name)

        # ------------------ Classical Bootstrap selection table (parallel) ------------------
        classical_boot_df = select_params_via_classical_bootstrap_parallel(
            bootstrap_paths,
            PARAM_TRIALS,
            signal_fn=signal_fn,
            utility_fn=UTILITY_FN,
            n_jobs=N_JOBS,
            backend="process"
        )
        selected_params['ERM_Boot'] = int(classical_boot_df.sort_values(utility_name, ascending=False).reset_index(drop=True).iloc[0]['param'])

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
        DESIRED_ORDER = [f"{p}th" for p in (10,20,30,40,50,60,70,80,90)] + ["ERM_Boot", "ERM_max"]
        ordered_names = [n for n in DESIRED_ORDER if n in selected_params]

        # ------------------ Evaluate train/test IN THIS ORDER ------------------
        records = []
        cumret_panels = {}
        train_len = len(train_returns)

        for name in ordered_names:
            L = selected_params[name]
            # Train
            pos_tr = signal_fn(train_returns, L)
            bt_tr_series = backtest_positions(train_returns, pos_tr)
            # Test (build positions on full series, slice to test)
            bt_te_series = apply_param_to_test(full_returns, train_len, L, signal_fn)

            # Metrics on scaled series
            m_tr = metrics(bt_tr_series)
            m_te = metrics(bt_te_series)

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
            full_bt_series  = backtest_positions(full_returns, full_pos)
            cum = (1 + full_bt_series).cumprod()
            cumret_panels[f"{name}_{L}"] = cum

        # Build DataFrame and lock the categorical order for plotting/printing
        results_df = pd.DataFrame(records)
        results_df["name"] = pd.Categorical(results_df["name"],
                                            categories=ordered_names,
                                            ordered=True)
        results_df = results_df.sort_values("name").reset_index(drop=True)
        results_df['asset'] = colname
        all_results.append(results_df)
    all_results_df = pd.concat(all_results, ignore_index=True)

    print("\n=== Parameter selections & metrics ===")

    # Save CSV
    results_path = os.path.join(outputs_path, "results", f"{SIGNAL_NAME}_individual_bootstrap_selection.csv")
    all_results_df.to_csv(results_path, index=False)

    # function to compute CI for a series
    def ci_normal(series, alpha=0.05):
        n = series.count()
        mean = series.mean()
        se = series.std(ddof=1) / np.sqrt(n)
        z = 1.96 if alpha == 0.05 else None
        lower = mean - z * se
        upper = mean + z * se
        return pd.Series({'mean': mean, 'lower': lower, 'upper': upper})

    filtered_all_results_df = all_results_df.loc[all_results_df['name'] != 'max']

    # input(filtered_all_results_df)

    # apply per group
    ci_df = filtered_all_results_df.groupby('name')[[f'{utility_name}_train', f'{utility_name}_test', f'Gap_{utility_name}']].apply(
        lambda df: df.apply(ci_normal)
    )

    # input(ci_df)

    # Slice rows by the 2nd index level
    ci_mean  = ci_df.xs('mean',  level=1)
    ci_lower = ci_df.xs('lower', level=1)
    ci_upper = ci_df.xs('upper', level=1)

    metrics = [f'{utility_name}_train', f'{utility_name}_test', f'Gap_{utility_name}']
    titles  = [f'{utility_name.capitalize()} Train (95% CI)', f'{utility_name.capitalize()} Test (95% CI)', f'Gap {utility_name.capitalize()} (95% CI)']

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    for ax, metric, title in zip(axes, metrics, titles):
        means  = ci_mean[metric]
        lowers = ci_lower[metric]
        uppers = ci_upper[metric]

        ax.errorbar(
            means.index,
            means.values,
            yerr=[(means - lowers).values, (uppers - means).values],
            fmt='o', capsize=5,
        )
        ax.set_ylabel(title)
        # ax.set_title(title)
        ax.legend()

    plt.xticks(rotation=90)
    plt.tight_layout()
    out = os.path.join(outputs_path, "results", f"cis-{SIGNAL_NAME}.png")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"\nSaved: {results_path}")
    print("Saved plots to:", os.path.join(outputs_path, "results"))

    # # ------------------ Plots ------------------
    # def plot_gap_local(df, metric_train, metric_test, gap_col, title, fname):
    #     x = df["name"].values
    #     fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)

    #     ax[0].plot(x, df[metric_train], marker="o", label="Train")
    #     ax[0].plot(x, df[metric_test], marker="o", label="Test")
    #     ax[0].set_ylabel(title)
    #     ax[0].grid(True, alpha=0.3)
    #     ax[0].legend()

    #     ax[1].plot(x, df[gap_col], marker="o", label="Generalization Gap")
    #     ax[1].set_xlabel("Selection")
    #     ax[1].set_ylabel("Gap")
    #     ax[1].grid(True, alpha=0.3)
    #     ax[1].legend()

    #     plt.xticks(rotation=30)
    #     out = os.path.join(outputs_path, "results", fname)
    #     fig.savefig(out, bbox_inches="tight", dpi=150)
    #     plt.close(fig)

    # # Generalization gap: Sharpe / Sortino / MaxDD / AvgDD
    # plot_gap_local(results_df, "Sharpe_train",  "Sharpe_test",  "Gap_Sharpe",  "Sharpe Ratio",         f"sharpe-gap-{SIGNAL_NAME}.png")
    # plot_gap_local(results_df, "Sortino_train", "Sortino_test", "Gap_Sortino", "Sortino Ratio",        f"sortino-gap-{SIGNAL_NAME}.png")
    # plot_gap_local(results_df, "MaxDD_train",   "MaxDD_test",   "Gap_MaxDD",   "Max Drawdown (%)",     f"maxdd-gap-{SIGNAL_NAME}.png")
    # plot_gap_local(results_df, "AvgDD_train",   "AvgDD_test",   "Gap_AvgDD",   "Average Drawdown (%)", f"avgdd-gap-{SIGNAL_NAME}.png")

    # # ------------------ Cumulative returns: Train vs Test panels (each starts at 1) ------------------
    # cum_df = pd.DataFrame(cumret_panels)  # full-sample cumrets (scaled) for each selection
    # test_start = test_returns.index[0]
    # train_start = train_returns.index[0]

    # # 1) TRAIN: rebuild incremental cumulative return from the underlying *daily* scaled returns
    # # Convert cum_df back to daily scaled returns (r_t = C_t / C_{t-1} - 1), then recompute
    # daily_rets = cum_df.pct_change().fillna(0.0)

    # # Train cumulative returns: only use train window and rebase to 1 at train_start
    # train_rets = daily_rets.loc[train_start:test_start].dropna()
    # train_cum = (1.0 + train_rets).cumprod()
    # # Ensure exact 1.0 at the first row
    # if not train_cum.empty:
    #     train_cum = train_cum / train_cum.iloc[0]

    # # 2) TEST: only use returns from test_start onward and rebase to 1 at test_start
    # test_rets = daily_rets.loc[test_start:].dropna()
    # test_cum = (1.0 + test_rets).cumprod()
    # if not test_cum.empty:
    #     test_cum = test_cum / test_cum.iloc[0]

    # # Plot
    # # ------------------ Distinct colors ------------------
    # palette = plt.get_cmap('tab20').colors  # 20 high-contrast colors
    # color_map = {}
    # for i, colname in enumerate(cum_df.columns):
    #     color_map[colname] = palette[i % len(palette)]

    # # ------------------ Build plots ------------------
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)

    # def plot_panel(ax, data, title):
    #     for col in data.columns:
    #         lw = 1.5
    #         zorder = 1
    #         alpha = 0.9
    #         # Highlight ERM_max
    #         if col.startswith("ERM_max"):
    #             lw = 2.5
    #             zorder = 3
    #             alpha = 1.0
    #         ax.plot(data.index, data[col],
    #                 label=col,
    #                 color=color_map[col],
    #                 linewidth=lw,
    #                 alpha=alpha,
    #                 zorder=zorder)
    #     # ax.set_title(title)
    #     ax.set_ylabel("Cumulative Return")
    #     # ax.grid(True, alpha=0.3)

    #     for ax in [ax1, ax2]:
    #         ax.tick_params(axis='x', labelsize=10)  # increase font size for x-axis
    #         # ax.tick_params(axis='y', labelsize=11)  # optional, y-axis too

    # # select 90th, 50th, and 10th percentiles for train/test panels + ERM_max
    # train_cum = train_cum.loc[:,train_cum.columns.str.contains("90|50|10|ERM_max")]
    # test_cum = test_cum.loc[:,test_cum.columns.str.contains("90|50|10|ERM_max")]

    # plot_panel(ax1, train_cum, "Train Period Cumulative Returns (rebased to 1)")
    # plot_panel(ax2, test_cum, "Test Period Cumulative Returns (rebased to 1)")

    # # ------------------ Single legend outside ------------------
    # handles, labels = ax1.get_legend_handles_labels()
    # fig.legend(handles,
    #            labels,
    #            loc='lower center',
    #            bbox_to_anchor=(0.5, -0.10),  # move legend lower
    #            ncol=len(labels),             # all entries in one row
    #            fontsize=13)

    # # Save
    # out = os.path.join(outputs_path, "results", f"cumret-{SIGNAL_NAME}.png")
    # fig.savefig(out, bbox_inches="tight", dpi=150)
    # plt.close(fig)

    # print(f"\nSaved: {results_path}")
    # print("Saved plots to:", os.path.join(outputs_path, "results"))
