import sys
import os

# sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import torch
import multiprocessing
import argparse
import copy

from settings import INPUT_PATH, OUTPUT_PATH
from signals.TSM import TSM
from estimators.DependentBootstrapSampling import DependentBootstrapSampling
from functionals.Functionals import Functionals
from portfolio_tools.Backtest import Backtest
from utils.conn_data import load_pickle, save_strat_opt_results

class training_etfstsm_moskowitz(TSM, DependentBootstrapSampling, Functionals):
    def __init__(self,
                 vol_target: float,
                 bar_name: str,
                 boot_method: str = "cbb",
                 Bsize: int = 100,
                 k: int = 100,
                 alpha: float=0.95,
                 utility: str="Sharpe",
                 use_seed: bool=True) -> None:
        """
        This class is a wrapper for the TSM class and the DependentBootstrapSampling class. 
        It is used to train the ETF TSM strategy.

        Parameters
        ----------
        vol_target : float
            The target volatility of the strategy.
        bar_name : str
            The name of the bar to use for the strategy.
        boot_method : str, optional
            The bootstrap method to use. The default is "cbb".
        Bsize : int, optional
            Block size to create the block set.
        k : int, optional
            The number of bootstrap samples to generate. The default is 100.
        alpha : float, optional
            The percentile to use for the functional. The default is 0.95.
        utility : str, optional
            The utility function to use. The default is "Sharpe".
        use_seed : int, optional
            If to use seed on the bootstraps or not. The default is None.

        Returns
        -------
        None.

        """

        Functionals.__init__(self, alpha=alpha)
    
        # init strategy attributes
        self.sysname = "training_etfstsm_moskowitz"
        self.instruments = [
        
            'SPY', 'IWM', 'EEM', 'TLT', 'USO', 'GLD', 'XLF',
            'XLB', 'XLK', 'XLV', 'XLI', 'XLU', 'XLY', 'XLP',
            'XLE', 'AGG', 'DBC', 'HYG', 'LQD','UUP'
        
        ]
        self.vol_target = vol_target
        self.bar_name = bar_name

        # inputs
        inputs = load_pickle(os.path.join(INPUT_PATH, self.sysname, f"{self.sysname}.pickle"))
        self.bars_info = inputs["bars"]

        # returns
        self.returns_info = self.build_returns()

        # carry
        self.carry_info = None

        # generate bootstrap samples from returns
        DependentBootstrapSampling.__init__(self,
                                            time_series=torch.tensor(self.returns_info.to_numpy()),
                                            boot_method=boot_method,
                                            Bsize=Bsize,
                                            use_seed=use_seed)
        self.all_samples = self.sample_many_paths(k=k)
        self.n_bootstrap_samples = self.all_samples.shape[0]

        # generate signals from bootstrap samples
        self.bootstrap_signals_info = None
        
        # generate forecasts from from bootstrap signals
        self.bootstrap_forecasts_info = None

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
            signal = self.Moskowitz(returns=self.returns_info[[f"{instrument}_returns"]], window=window)
            signals[instrument] = signal.rename(columns={f"{instrument}_returns": self.bar_name})
            
        return signals
    
    def build_forecasts(self):
        forecasts = {}
        for instrument in self.instruments:
            forecast = np.where(self.signals_info[instrument][[self.bar_name]] > 0, 1, -1)

            forecasts[instrument] = pd.DataFrame(forecast,
                                                 index=self.signals_info[instrument].index,
                                                 columns=[self.bar_name])
            
        return forecasts
    
    def build_signals_from_bootstrap_samples(self, window: int):
        bootrap_signals = {}
        for i in range(self.n_bootstrap_samples):
            signals = {}
            sample_df = pd.DataFrame(self.all_samples[i, :, :], columns=self.instruments, index=self.returns_info.index)
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
    strategy_params = params['strategy_params']
    window = params['window']

    # Initialize strategy within each process
    local_strategy = training_etfstsm_moskowitz(
        vol_target=strategy_params['vol_target'],
        bar_name=strategy_params['bar_name'],
        boot_method=strategy_params['boot_method'],
        Bsize=strategy_params['Bsize'],
        k=strategy_params['k'],
        alpha=strategy_params['alpha'],
        utility=strategy_params['utility'],
        use_seed=strategy_params['use_seed'])

    # for a given window, build signals from bootstrap samples
    local_strategy.bootstrap_signals_info = local_strategy.build_signals_from_bootstrap_samples(window=window)

    # build forecasts from bootstrap signals
    local_strategy.bootstrap_forecasts_info = local_strategy.build_forecasts_from_bootstrap_signals()

    # run backtest for each boostrap samples
    utilities_given_hyperparam = []
    for i in range(local_strategy.n_bootstrap_samples):
        # build signals info
        local_strategy.signals_info = local_strategy.bootstrap_signals_info[f"bootstrap_{i}"]

        # build forecasts info
        local_strategy.forecasts_info = local_strategy.bootstrap_forecasts_info[f"bootstrap_{i}"]

        # run backtest
        cerebro = Backtest(strat_metadata=local_strategy)
        cerebro.run_backtest(start_date=strategy_params['start_date'],
                             end_date=strategy_params['end_date'],
                             instruments=local_strategy.instruments,
                             bar_name=local_strategy.bar_name,
                             vol_window=252,
                             vol_target=local_strategy.vol_target,
                             resample_freq="B")
        
        # compute strategy performance
        metrics = cerebro.compute_summary_statistics(portfolio_returns=cerebro.agg_scaled_portfolio_returns)
        utilities_given_hyperparam.append(metrics[local_strategy.utility])

    return (torch.tensor(utilities_given_hyperparam))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--utility', type=str, help='Utility for the strategy returns evaluation.', default="Sharpe")
    parser.add_argument('--functional', type=str, help='Functional to aggregate across bootstrap samples.', default="means")
    parser.add_argument('--alpha', type=float, help='Percentile of the empirical distribution.', default=1) # -1 = minimum, 1 = maximum
    parser.add_argument('--k', type=int, help='Number of bootstrap samples.', default=10)
    parser.add_argument('--cpu_count', type=int, help='Number of CPUs to parallelize process.', default=1)
    parser.add_argument('--start_date', type=str, help='Start date for the strategy.', default=None)
    parser.add_argument('--end_date', type=str, help='End date for the strategy.', default="2015-12-31")
    parser.add_argument('--use_seed', type=int, help='If to use seed on the bootstraps or not.', default=True)

    args = parser.parse_args()

    if args.cpu_count == -1:
        args.cpu_count = multiprocessing.cpu_count() - 1

    # define the parameters for strategy initialization
    strategy_params = {
            'start_date': args.start_date,
            'end_date': args.end_date,
            'vol_target': 0.15,
            'bar_name': "Close",
            'boot_method': "cbb",
            'Bsize': 100,
            'k': args.k,
            'alpha': args.alpha,
            'utility': args.utility,
            'functional': args.functional,
            'use_seed': args.use_seed
    }

    # define parameters list for multiprocessing
    windows = range(30, 252 + 1, 1)
    parameters_list = [
        {
            'strategy_params': strategy_params,
            'window': w
        } for w in windows
    ]

    # define multiprocessing pool
    utilities = []
    with multiprocessing.Pool(processes=args.cpu_count) as pool:
        utilities = pool.map(objective, parameters_list)

    # final strategy inputs
    strategy = training_etfstsm_moskowitz(vol_target=strategy_params['vol_target'],
                                bar_name=strategy_params['bar_name'],
                                k=strategy_params['k'],
                                alpha=strategy_params['alpha'],
                                utility=strategy_params['utility'],
                                use_seed=strategy_params['use_seed'])
        
    # applying the functional
    final_utility = strategy.apply_functional(x=utilities, func=args.functional)

    # find position of scores that match final_utility
    position = strategy.find_utility_position(utilities=utilities, utility_value=final_utility)

    # find window that matches position
    robust_parameter = windows[position]

    # save relevant attributes fro optimization
    strategy.utilities = utilities
    strategy.windows = windows
    strategy.final_utility = final_utility
    strategy.robust_parameter = robust_parameter

    # results path
    results_path = os.path.join(OUTPUT_PATH, strategy.sysname, f'{args.utility}_{args.functional}_{args.alpha}_{args.k}')

    # run strategy with robust parameter IN-SAMPLE
    strategy.signals_info = strategy.build_signals(window=robust_parameter)
    strategy.forecasts_info = strategy.build_forecasts()
    cerebro = Backtest(strat_metadata=strategy)
    cerebro.run_backtest(start_date=args.start_date,
                         end_date=args.end_date,
                         instruments=strategy.instruments,
                         bar_name=strategy.bar_name,
                         vol_window=90,
                         vol_target=strategy.vol_target,
                         resample_freq="B")
    
    train_cerebro = copy.deepcopy(cerebro)
    save_strat_opt_results(results_path=results_path,
                           args=args,
                           cerebro=train_cerebro,
                           strategy=strategy,
                           train=True)
     
    # run strategy with robust parameter OUT-OF-SAMPLE
    strategy.signals_info = strategy.build_signals(window=robust_parameter)
    strategy.forecasts_info = strategy.build_forecasts()
    cerebro = Backtest(strat_metadata=strategy)
    cerebro.run_backtest(start_date=args.end_date,
                         end_date=None,
                         instruments=strategy.instruments,
                         bar_name=strategy.bar_name,
                         vol_window=90,
                         vol_target=strategy.vol_target,
                         resample_freq="B")
    
    test_cerebro = copy.deepcopy(cerebro)   
    save_strat_opt_results(results_path=results_path,
                           args=args,
                           cerebro=test_cerebro,
                           strategy=strategy,
                           train=False)
    
    print(f"Optimization results saved in {results_path}")



    

        


