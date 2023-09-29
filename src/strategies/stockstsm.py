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
    def __init__(self, simulation_start, vol_target, bar_names) -> None:
        self.sysname = "stockstsm"
        self.instruments = ["AA", "ABM", "ABT"]
        self.simulation_start = simulation_start
        self.vol_target = vol_target
        self.bar_names = bar_names

        # inputs
        self.bars_info = load_pickle(os.path.join(INPUT_PATH, "crsp_nyse.pickle"))
        self.signals_info = self.build_signals()
        self.forecasts_info = self.build_forecasts()

        # outputs
        if os.path.exists(os.path.join(OUTPUT_PATH, self.sysname, "{}.pickle".format(self.sysname))):
            self.strat_outputs = load_pickle(path=os.path.join(OUTPUT_PATH, self.sysname, "{}.pickle".format(self.sysname)))
        else:
            self.strat_outputs = None

    def build_signals(self):
        signals_info = {}
        for inst in self.instruments:
            bars = self.bars_info[inst][self.bar_names].resample("B").last().ffill()
            signals = self.Moskowitz(prices=bars, window=252)

            signals.rename(columns={"curAdjClose": "{} signals".format(inst)}, inplace=True)

            signals_info[inst] = signals
        
        return signals_info

    def build_forecasts(self):
        forecasts_info = {}
        for inst in list(self.signals_info.keys()):
            signals = self.signals_info[inst]
            tmp_forecasts = pd.DataFrame(np.where(signals > 0, 1, -1),
                                        columns=["{} forecasts".format(inst)],
                                        index=signals.index)
            
            forecasts_info[inst] = tmp_forecasts
        
        return forecasts_info

if __name__ == "__main__":
    strat_metadata = stockstsm(simulation_start=None, vol_target=0.2, bar_names=["curAdjClose"])

    cerebro = Backtest(strat_metadata=strat_metadata)

    portfolio_df = cerebro.run_backtest(instruments=strat_metadata.instruments,
                                        bar_name="Close",
                                        vol_window=90,
                                        vol_target=strat_metadata.vol_target,
                                        resample_freq="B",
                                        capital=10000,
                                        reinvest=False)

    strat_metadata.strat_inputs["portfolio"] = portfolio_df
    output_path = os.path.join(OUTPUT_PATH, strat_metadata.sysname, "{}.pickle".format(strat_metadata.sysname))
    save_pickle(obj=strat_metadata.strat_inputs, path=output_path)