import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np

from settings import INPUT_PATH, OUTPUT_PATH
from signals.TSM import TSM
from portfolio_tools.Backtest import Backtest
from utils.conn_data import load_pickle, save_pickle
from data.ETFsLoader import ETFsLoader

class training_etfstsm(TSM):
    def __init__(self, simulation_start, vol_target, bar_name) -> None:
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

        # signals
        self.signals = None

        # outputs
        if os.path.exists(os.path.join(OUTPUT_PATH, self.sysname, f"{self.sysname}.pickle")):
            self.strat_outputs = load_pickle(path=os.path.join(OUTPUT_PATH, self.sysname, f"{self.sysname}.pickle"))
        else:
            self.strat_outputs = None
            
    def build_signals(self, window):
        signals = {}
        for instrument in self.instruments:
            signal = self.Moskowitz(prices=self.bars_info[instrument][[self.bar_name]], window=window)
            signals[instrument] = signal.rename(columns={self.bar_name: f"{instrument}_signals"})
            
        return signals
    
    def build_forecasts(self):
        forecasts = {}
        for instrument in self.instruments:
            forecast = np.where(self.signals_info[instrument][[f"{instrument}_signals"]] > 0, 1, -1)

            forecasts[instrument] = pd.DataFrame(forecast,
                                                 index=self.signals_info[instrument].index,
                                                 columns=[f"{instrument}_forecasts"])
            
        return forecasts

if __name__ == "__main__":
    
    # strategy inputs
    strategy = training_etfstsm(simulation_start=None, vol_target=0.2, bar_name="Close")

    # strategy hyperparameters
    windows = range(30, 252 + 1, 1)

    # strategy optimization
    for w in windows:
        strategy.signals_info = strategy.build_signals(window=w)
        strategy.forecasts_info = strategy.build_forecasts()
        cerebro = Backtest(strat_metadata=strategy)
        cerebro.run()

