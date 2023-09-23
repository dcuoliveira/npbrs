import os
import pandas as pd
import numpy as np

from settings import INPUT_PATH, OUTPUT_PATH
from src.signals.TSM import TSM
from portfolio_tools.Backtest import Backtest
from src.utils.conn_data import load_pickle, save_pickle

class stockstsm(TSM):
    def __init__(self, simulation_start, vol_target) -> None:
        self.sysname = "fxmmts"
        G10 = ["USDEUR", "USDJPY", "USDAUD", "USDNZD", "USDCAD", "USDGBP", "USDCHF", "USDSEK", "USDNOK"]
        LATAM = ['WDO1', 'USDCLP', 'USDZAR', 'USDMXN', "USDCOP"]
        C3 = ["USDHUF", "USDPLN", "USDCZK"]
        ASIA = ["USDCNH", "USDTWD", "USDINR", "USDKRW"]
        self.instruments = G10 + LATAM + C3 + ASIA
        self.simulation_start = simulation_start
        self.vol_target = vol_target

        # inputs
        self.strat_inputs = load_pickle(os.path.join(INPUT_PATH, self.sysname, "{}.pickle".format(self.sysname)))
        self.bars_info = self.strat_inputs["bars"]
        self.carry_info = self.strat_inputs["carry"]
        self.signals_info = self.strat_inputs["signals"]
        self.forecasts_info = self.build_forecasts()

        # outputs
        if os.path.exists(os.path.join(OUTPUT_PATH, self.sysname, "{}.pickle".format(self.sysname))):
            self.strat_outputs = load_pickle(path=os.path.join(OUTPUT_PATH, self.sysname, "{}.pickle".format(self.sysname)))
        else:
            self.strat_outputs = None

    def build_forecasts(self):
        forecasts_info = {}
        for inst in list(self.signals_info.keys()):
            tmp_signals = self.signals_info[inst].resample("B").last().ffill().mean(axis=1)

            tmp_forecasts = pd.DataFrame(np.where(tmp_signals > 0, 1, -1),
                                        columns=["{} forecasts".format(inst)],
                                        index=tmp_signals.index)
            forecasts_info[inst] = tmp_forecasts
        
        return forecasts_info

if __name__ == "__main__":
    strat_metadata = stockstsm(simulation_start=None, vol_target=0.2)

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