import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import torch

from settings import INPUT_PATH, OUTPUT_PATH
from signals.TSM import TSM
from estimators.DependentBootstrapSampling import DependentBootstrapSampling
from portfolio_tools.Backtest import Backtest
from utils.conn_data import load_pickle, save_pickle

class training_etfstsm(TSM, DependentBootstrapSampling):
    def __init__(self,
                 simulation_start: str,
                 vol_target: float,
                 bar_name: str,
                 boot_method: str = "cbb",
                 Bsize: int = 100,
                 k: int = 100) -> None:
    
        # init strategy attributes
        self.sysname = "training_etfstsm"
        self.instruments = [
        
            'SPY', 'IWM', 'EEM', 'TLT', 'USO', 'GLD', 'XLF',
            'XLB', 'XLK', 'XLV', 'XLI', 'XLU', 'XLY', 'XLP',
            'XLE', 'VIX', 'AGG', 'DBC', 'HYG', 'LQD','UUP'
        
        ]
        self.simulation_start = simulation_start
        self.vol_target = vol_target
        self.bar_name = bar_name

        # inputs
        inputs = load_pickle(os.path.join(INPUT_PATH, self.sysname, f"{self.sysname}.pickle"))
        self.bars_info = inputs["bars"]

        # returns
        self.returns = self.build_returns()

        # carry
        self.carry_info = None

        # generate bootstrap samples from returns
        DependentBootstrapSampling.__init__(self,
                                            time_series=torch.tensor(self.returns.to_numpy()),
                                            boot_method=boot_method,
                                            Bsize=Bsize)
        self.all_samples = self.sample_many_paths(k=k)
        self.n_bootstrap_samples = self.all_samples.shape[0]

        # generate signals from bootstrap samples
        self.bootstrap_signals_info = None
        
        # generate forecasts from from bootstrap signals
        self.bootstrap_forecasts_info = None

        # check if dir exists
        if not os.path.exists(os.path.join(OUTPUT_PATH, self.sysname)):
            os.makedirs(os.path.join(OUTPUT_PATH, self.sysname))

        # export outputs
        if os.path.exists(os.path.join(OUTPUT_PATH, self.sysname, f"{self.sysname}.pickle")):
            self.strat_outputs = load_pickle(path=os.path.join(OUTPUT_PATH, self.sysname, f"{self.sysname}.pickle"))
        else:
            self.strat_outputs = None

    def build_returns(self):
        returns = []
        for instrument in self.instruments:
            tmp_return = np.log(self.bars_info[instrument][[self.bar_name]]).diff().dropna()
            returns.append(tmp_return.rename(columns={self.bar_name: f"{instrument}_returns"}))

        returns_df = pd.concat(returns, axis=1)
            
        return returns_df
            
    def build_signals(self, window: int):
        signals = {}
        for instrument in self.instruments:
            signal = self.Moskowitz(prices=self.bars_info[instrument][[self.bar_name]], window=window)
            signals[instrument] = signal
            
        return signals
    
    def build_forecasts(self):
        forecasts = {}
        for instrument in self.instruments:
            forecast = np.where(self.signals_info[instrument][[f"{instrument}_signals"]] > 0, 1, -1)

            forecasts[instrument] = pd.DataFrame(forecast,
                                                 index=self.signals_info[instrument].index,
                                                 columns=[self.bar_name])
            
        return forecasts
    
    def build_signals_from_bootstrap_samples(self, window: int):
        bootrap_signals = {}
        for i in range(self.n_bootstrap_samples):
            signals = {}
            sample_df = pd.DataFrame(self.all_samples[i, :, :], columns=self.instruments, index=self.returns.index)
            for instrument in self.instruments:
                signal = self.Moskowitz(returns=sample_df[[instrument]], window=window)
                signals[instrument] = signal.rename(columns={instrument: self.bar_name})
            
            bootrap_signals[f"bootstrap_{i}"] = signals

        return bootrap_signals
    
    def build_forecasts_from_bootstrap_signals(self):
            bootrap_forecasts = {}
            for i in range(self.n_bootstrap_samples):
                forecasts = {}
                for instrument in self.instruments:
                    forecast = np.where(self.bootstrap_signals_info[f"bootstrap_{i}"][instrument][[self.bar_name]] > 0, 1, -1)
    
                    forecasts[instrument] = pd.DataFrame(forecast,
                                                         index=self.bootstrap_signals_info[f"bootstrap_{i}"][instrument].index,
                                                         columns=[self.bar_name])
                
                bootrap_forecasts[f"bootstrap_{i}"] = forecasts
    
            return bootrap_forecasts

if __name__ == "__main__":
    
    # strategy inputs
    strategy = training_etfstsm(simulation_start=None, vol_target=0.2, bar_name="Close")

    # strategy hyperparameters
    windows = range(30, 252 + 1, 1)

    # strategy optimization
    for w in windows:
        # for a given window, build signals from bootstrap samples
        strategy.bootstrap_signals_info = strategy.build_signals_from_bootstrap_samples(window=w)

        # build forecasts from bootstrap signals
        strategy.bootstrap_forecasts_info = strategy.build_forecasts_from_bootstrap_signals()

        # run backtest for each boostrap samples
        for i in range(strategy.n_bootstrap_samples):
            # build signals info
            strategy.signals_info = strategy.bootstrap_signals_info[f"bootstrap_{i}"]

            # build forecasts info
            strategy.forecasts_info = strategy.bootstrap_forecasts_info[f"bootstrap_{i}"]

            # run backtest
            cerebro = Backtest(strat_metadata=strategy)
            cerebro.run_backtest(instruments=strategy.instruments,
                                 bar_name=strategy.bar_name,
                                 vol_window=252,
                                 vol_target=strategy.vol_target,
                                 resample_freq="B")
            
            # compute strategy performance
            utilities = cerebro.compute_summary_statistics(portfolio_returns=cerebro.agg_scaled_portfolio_returns)

            


