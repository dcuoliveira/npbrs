import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import torch
import multiprocessing
import copy
import argparse

from settings import INPUT_PATH, OUTPUT_PATH
from signals.TSM import TSM
from estimators.DependentBootstrapSampling import DependentBootstrapSampling
from functionals.Functionals import Functionals
from portfolio_tools.Backtest import Backtest
from utils.conn_data import load_pickle, save_strat_opt_results

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
        self.returns_info = self.build_returns()

        # carry
        self.carry_info = None

        # generate bootstrap samples from returns
        DependentBootstrapSampling.__init__(self,
                                            time_series=torch.tensor(self.returns_info.to_numpy()),
                                            boot_method=boot_method,
                                            Bsize=Bsize)
        self.all_samples = self.sample_many_paths(k=k)
        self.n_bootstrap_samples = self.all_samples.shape[0]

        # generate signals from bootstrap samples
        self.bootstrap_signals_info = None
        
        # generate forecasts from from bootstrap signals
        self.bootstrap_forecasts_info = None

        # utilities
        self.utility = utility

    
def objective(params):
    # Extract the strategy and window from params
    strategy = params['strategy']

    # run backtest for each boostrap samples
    utilities_given_hyperparam = []
    for i in range(strategy.n_bootstrap_samples):
        utilities_given_hyperparam.append(i)

    return (torch.tensor(utilities_given_hyperparam))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--utility', type=str, help='Utility for the strategy returns evaluation.', default="Sharpe")
    parser.add_argument('--functional', type=str, help='Functional to aggregate across bootstrap samples.', default="means")
    parser.add_argument('--alpha', type=float, help='Confidence level for the rank of the estimates.', default=0.95)
    parser.add_argument('--k', type=int, help='Number of bootstrap samples.', default=10)
    parser.add_argument('--cpu_count', type=int, help='Number of CPUs to parallelize process.', default=-1)

    args = parser.parse_args()

    if args.cpu_count == -1:
        args.cpu_count = multiprocessing.cpu_count() - 1

    # strategy inputs
    strategy = training_etfstsm(simulation_start=None,
                                vol_target=0.2,
                                bar_name="Close",
                                k=args.k,
                                alpha=args.alpha,
                                utility=args.utility,
                                functional=args.functional)

    # strategy hyperparameters
    windows = range(30, 252 + 1, 1)

    # define multiprocessing pool``
    utilities = []

    with multiprocessing.Pool(processes=args.cpu_count) as pool:

        # define parameters list for the objective
        parameters_list = [{'strategy': copy.deepcopy(strategy), 'window': w} for w in windows]

        utilities = pool.map(objective, parameters_list)
        



        

            


