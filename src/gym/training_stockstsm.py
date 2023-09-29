import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from settings import INPUT_PATH, OUTPUT_PATH
from signals.TSM import TSM
from portfolio_tools.Backtest import Backtest
from utils.conn_data import load_pickle, save_pickle

class stockstsm(TSM):
    def __init__(self, simulation_start, vol_target, bar_name) -> None:
        self.sysname = "stockstsm"
        self.instruments = ["AA", "ABM", "ABT"]
        self.simulation_start = simulation_start
        self.vol_target = vol_target
        self.bar_name = bar_name

        # inputs
        self.bars_info = load_pickle(os.path.join(INPUT_PATH, "crsp_nyse.pickle"))
        self.signals_info = self.build_signals()
        self.forecasts_info = self.build_forecasts()

        # outputs
        if os.path.exists(os.path.join(OUTPUT_PATH, self.sysname, "{}.pickle".format(self.sysname))):
            self.strat_outputs = load_pickle(path=os.path.join(OUTPUT_PATH, self.sysname, "{}.pickle".format(self.sysname)))
        else:
            self.strat_outputs = None

if __name__ == "__main__":
    
    # strategy inputs
    strat_metadata = stockstsm(simulation_start=None, vol_target=0.2, bar_name="curAdjClose")

    # strategy hyperparameters
    windows = range(30, 252 + 1, 1)

    # strategy optimization
    for w in windows:
        strat_metadata.signals_info = strat_metadata.build_signals(window=w)
        strat_metadata.forecasts_info = strat_metadata.build_forecasts()

        cerebro = Backtest(strat_metadata=strat_metadata)
        cerebro.run()
