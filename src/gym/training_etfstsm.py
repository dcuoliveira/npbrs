import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import torch
import multiprocessing
import copy

from settings import INPUT_PATH, OUTPUT_PATH
from signals.TSM import TSM
from estimators.DependentBootstrapSampling import DependentBootstrapSampling
from functionals.Functionals import Functionals
from portfolio_tools.Backtest import Backtest
from utils.conn_data import load_pickle

class training_etfstsm(TSM, DependentBootstrapSampling, Functionals):
    def __init__(self,
                 simulation_start: str,
                 vol_target: float,
                 bar_name: str,
                 boot_method: str = "cbb",
                 Bsize: int = 100,
                 k: int = 100,
                 alpha: float=0.95,
                 utility: str="Sharpe",
                 functional: str="means") -> None:
        """
        This class is a wrapper for the TSM class and the DependentBootstrapSampling class. 
        It is used to train the ETF TSM strategy.

        Parameters
        ----------
        simulation_start : str
            The date from which to start the simulation.
        vol_target : float
            The target volatility of the strategy.
        bar_name : str
            The name of the bar to use for the strategy.
        boot_method : str, optional
            The bootstrap method to use. The default is "cbb".
        Bsize : int, optional
            The size of the bootstrap samples. The default is 100.
        k : int, optional
            The number of bootstrap samples to generate. The default is 100.
        alpha : float, optional
            The percentile to use for the functional. The default is 0.95.
        utility : str, optional
            The utility function to use. The default is "Sharpe".
        functional : str, optional
            The functional to use. The default is "means".

        Returns
        -------
        None.

        """

        Functionals.__init__(self, alpha=alpha)
    
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
        # utilities
        self.utility = utility

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
    
def objective(params):
    strategy = params["strategy"]
    window = params["window"]

    # for a given window, build signals from bootstrap samples
    strategy.bootstrap_signals_info = strategy.build_signals_from_bootstrap_samples(window=window)

    # build forecasts from bootstrap signals
    strategy.bootstrap_forecasts_info = strategy.build_forecasts_from_bootstrap_signals()

    # run backtest for each boostrap samples
    utilities_given_hyperparam = []
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
        metrics = cerebro.compute_summary_statistics(portfolio_returns=cerebro.agg_scaled_portfolio_returns)
        utilities_given_hyperparam.append(metrics[strategy.utility])

    return (torch.tensor(utilities_given_hyperparam))

if __name__ == "__main__":
    
    utility = "Sharpe"
    functional = "means"
    alpha = 0.95

    # strategy inputs
    strategy = training_etfstsm(simulation_start=None,
                                vol_target=0.2,
                                bar_name="Close",
                                k=10,
                                alpha=alpha,
                                utility=utility,
                                functional=functional)

    # strategy hyperparameters
    # windows = range(30, 252 + 1, 1)
    windows = range(30, 35 + 1, 1)
    cpu_count = 4 # multiprocessing.cpu_count()

    # define multiprocessing pool
    utilities = []

    with multiprocessing.Pool(processes=cpu_count) as pool:

        # define parameters list for the objective
        parameters_list = [{'strategy': copy.deepcopy(strategy), 'window': w} for w in windows]

        utilities = pool.map(objective,parameters_list)
        
    # applying the functional
    final_utility = strategy.apply_functional(x=utilities, func=functional)

    # find position of scores that match final_utility
    position = strategy.find_utility_position(utilities=utilities, utility_value=final_utility)

    # find window that matches position
    robust_parameter = windows[position]
    print(robust_parameter)


        

            


