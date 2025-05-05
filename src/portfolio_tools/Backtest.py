import pandas as pd
import numpy as np

from utils.conn_data import load_pickle, save_pickle
from settings import INPUT_PATH
from portfolio_tools.Diagnostics import Diagnostics

class Backtest(Diagnostics):
    def __init__(self, strat_metadata: classmethod) -> None:
        super().__init__()

        self.bars = strat_metadata.bars_info
        self.signals = strat_metadata.signals_info
        self.forecasts = strat_metadata.forecasts_info
        self.carry = strat_metadata.carry_info
        self.train_size = strat_metadata.train_size
        self.data_freq = strat_metadata.data_freq

    def standardize_inputs(self,
                           start_date: str,
                           end_date: str,
                           train_flag: bool,
                           instruments,
                           bar_name: str,
                           vol_window: int,
                           resample_freq: str):

        bars_list = []
        vols_list = []
        rets_list = []
        carrys_list = []
        signals_list = []
        forecasts_list = []
        for inst in instruments:
            tmp_bars = self.bars[inst][[bar_name]].resample(resample_freq).last().ffill()

            tmp_rets = np.log(tmp_bars).diff()
            tmp_vols = tmp_rets.rolling(window=vol_window).std() * np.sqrt(self.data_freq)

            tmp_signals = self.signals[inst].resample(resample_freq).last().ffill()
            tmp_forecasts = self.forecasts[inst].resample(resample_freq).last().ffill()

            if self.carry is None:
                tmp_carry = tmp_forecasts.copy().rename(columns={"{} forecasts".format(inst): bar_name})
                tmp_carry[bar_name] = 0
            else:
                tmp_carry = self.carry[inst][[bar_name]].resample(resample_freq).last().ffill()

            bars_list.append(tmp_bars.rename(columns={bar_name: inst}))
            vols_list.append(tmp_vols.rename(columns={bar_name: inst}))
            rets_list.append(tmp_rets.rename(columns={bar_name: inst}))
            carrys_list.append(tmp_carry.rename(columns={bar_name: inst}))
            signals_list.append(tmp_signals.rename(columns={bar_name: inst}))
            forecasts_list.append(tmp_forecasts.rename(columns={bar_name: inst}))

        self.bars_df = pd.concat(bars_list, axis=1)
        self.vols_df = pd.concat(vols_list, axis=1)
        self.rets_df = pd.concat(rets_list, axis=1)
        self.carrys_df = pd.concat(carrys_list, axis=1)
        self.signals_df = pd.concat(signals_list, axis=1)
        self.forecasts_df = pd.concat(forecasts_list, axis=1)
                
        if (end_date is None) and (self.train_size is not None):

            if train_flag is None:
                raise ValueError("train_flag must be specified if end_date is None")
            end_size = int(len(self.signals_df) * self.train_size)

            if train_flag:
                start_date = self.signals_df.index[0]
                end_date = tmp_bars.index[end_size]
            else:
                start_date = tmp_bars.index[end_size]
                end_date = tmp_bars.index[-1]
        elif (end_date is None) and (self.train_size is None):
            raise ValueError("end_date and train_size cannot both be None")
        else:
            print("end_date is not None, using end_date as specified")

        self.bars_df = self.bars_df.loc[start_date:end_date]
        self.vols_df = self.vols_df.loc[start_date:end_date]
        self.rets_df = self.rets_df.loc[start_date:end_date]
        self.carrys_df = self.carrys_df.loc[start_date:end_date]
        self.signals_df = self.signals_df.loc[start_date:end_date]
        self.forecasts_df = self.forecasts_df.loc[start_date:end_date]

    def run_backtest(self,
                     instruments: list,
                     bar_name: str,
                     vol_window: int,
                     vol_target: float,
                     resample_freq: str,
                     start_date: str,
                     end_date: str,
                     train_flag: bool=None,):

        # standardize dict inputs
        self.standardize_inputs(start_date=start_date,
                                end_date=end_date,
                                train_flag=train_flag,
                                instruments=instruments,
                                bar_name=bar_name,
                                vol_window=vol_window,
                                resample_freq=resample_freq)
        
        # compute vol scaling
        vol_scale = (vol_target / self.vols_df).shift(1)

        # compute portfolio returns
        self.portfolio_returns = (self.forecasts_df * self.rets_df.shift(-1)).fillna(0)

        # compute scaled portfolio returns
        self.scaled_portfolio_returns = vol_scale * self.portfolio_returns

        # aggregate portfolio returns
        self.agg_portfolio_returns = pd.DataFrame(self.portfolio_returns.mean(axis=1), columns=["portfolio_returns"])
        self.agg_scaled_portfolio_returns = pd.DataFrame(self.scaled_portfolio_returns.mean(axis=1), columns=["portfolio_returns"])


        
