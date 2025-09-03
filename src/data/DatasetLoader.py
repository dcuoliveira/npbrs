import os
import pandas as pd
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

DEBUG = False

if DEBUG:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.dataset_utils import futures_dictionary_df
from utils.conn_data import load_pickle

asset_classes = {
    'commodities': [
        'C_', 'ZC', 'O_', 'ZO', 'S_', 'ZS', 'SM', 'ZM', 'BO', 'ZL', # grains
        'W_', 'ZW', 'KW', 'MW', 'NR', 'ZR', # grains
        'FC', 'LC', 'LH', 'PB', 'DA', # meats
        'CC', 'KC', 'SB', 'JO', 'CT', 'LB', # foodfibr
        'CL', 'ZU', 'RB', 'ZB', 'HO', 'ZH', 'NG', 'ZN', 'BC', 'BG', # oils
        'GC', 'ZG', 'SI', 'ZI', 'PA', 'PL', 'HG', # metals
    ],
    'equities': [
        'AX', 'LX', 'CA', 'MD', 'CR', 'ND', 'DJ', 'NK', 'DX', 'SC', 'EN', 'SP',
        'ER', 'RL', 'ES', 'XU', 'GI', 'XX', 'HS', 'YM', 'XX', 'AP',
    ],
    'rates': [
        'ED', 'FF', 'CB', 'US', 'TY', 'FB', 'UA', 'TA', 'DT', 'GS', 'SS',
    ],
    'currencies': [
        'BN', 'CN', 'FX', 'FN', 'JN', 'SF', 'SN', 'AD', 'AN', 'MP',
    ]
}

expirations = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']
month_map = {code: i + 1 for i, code in enumerate(expirations)}

class DatasetLoader(object):
    """
    
    """
    
    def __init__(self, flds):
        self.inputs_path = os.path.join(os.path.dirname(__file__), 'inputs')
        self.flds = flds
        self.ticker_encoder = LabelEncoder()

    @staticmethod
    def forward_fill_until_last(series):
        result = series.copy()
        last_valid_index = series.last_valid_index()
        if last_valid_index is None:
            return result
        mask = series.index <= last_valid_index
        result.loc[mask] = series[mask].ffill()
        return result

    @staticmethod
    def compute_transition_matrix(data: pd.DataFrame):
        output = {}
        for i in range(data.columns.size):
            cur_data = data[f"cluster_step{i}"]
            cur_date = cur_data.dropna().index[-1].strftime("%Y-%m-%d")
            transition_matrix = np.zeros((6, 6))
            for j in range(len(cur_data) - 1):
                cur = cur_data.iloc[j]
                nex = cur_data.iloc[j + 1]
                if np.isnan(nex):
                    break
                transition_matrix[int(cur), int(nex)] += 1
            transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
            output[cur_date] = transition_matrix.copy()
            
        return output

    def build_futures_summary_data(self, continuous_future_method):
        target_path = os.path.join(self.inputs_path, 'pinnacle', 'CLCDATA')
        selected_files = [file_name for file_name in os.listdir(target_path) if ('.CSV' in file_name) and (f'_{continuous_future_method}' in file_name)]
        data = []
        for file_name in selected_files:
            tmp_data = pd.read_csv(os.path.join(target_path, file_name))
            ticker = file_name.split('_')[0]
            tmp_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'open int']
            tmp_data['date'] = pd.to_datetime(tmp_data['date'], format='%m/%d/%Y')
            melt_tmp_data = tmp_data.melt('date')
            melt_tmp_data['ticker'] = ticker
            melt_tmp_data.rename(columns={'variable': 'flds'}, inplace=True)
            melt_tmp_data = melt_tmp_data[['date', 'ticker', 'flds', 'value']]
            data.append(melt_tmp_data)
        data = pd.concat(data, axis=0)
        return data

    def generic_melted_loader(self, dataset_name, data=None):
        if data is None:
            orig_data = pd.read_excel(os.path.join(self.inputs_path, 'bbg', f'melted_{dataset_name}.xlsx'), engine='openpyxl')
        else:
            orig_data = deepcopy(data)

        data = []
        for ticker in self.flds.keys():
            selected_flds = self.flds[ticker]
            for fld in selected_flds:
                tmp_data = orig_data.loc[(orig_data['ticker'] == ticker) & (orig_data['flds'] == fld)]
                tmp_data_pivot = tmp_data.pivot_table(index=['date'], columns=['ticker'], values=['value'])
                tmp_data_pivot.columns = tmp_data_pivot.columns.droplevel()
                data.append(tmp_data_pivot)
        data = pd.concat(data, axis=1)
        return data

    def stock_bond_optmization_data(self, dataset_name, freq):
        data = self.generic_melted_loader(dataset_name)

        # resample data
        data = data.resample("D").last().ffill()
        data = data.resample(freq).last()

        # compute us treasury total returns
        data['LUATTRUU Index'] = np.log(data['LUATTRUU Index']).diff()

        # divide spx returns by 100
        data['SPX Index'] = data['SPX Index'] / 100

        # dropna
        data = data.dropna()

        # filter date
        data = data.loc['2000-01-01':]
        data = data.sort_index(axis=1)

        self.dt_index = data.index

        return data
    
    def zzr_etfs_optmization_data(self, dataset_name):
        data = self.generic_melted_loader(dataset_name)

        # resample data
        data = data.resample('B').last().ffill()

        # compute log returns
        data = np.log(data).diff()

        # dropna
        data = data.dropna()
        data = data.sort_index(axis=1)

        self.dt_index = data.index

        return data
    
    def lzr_tsmom_data(self,
                    dataset_name,
                    continuous_future_method,
                    start_year,
                    end_year,
                    categorical_tickers):

        data = self.build_futures_summary_data(continuous_future_method)
        data = self.generic_melted_loader(dataset_name, data)
        data = data.resample('B').last().ffill()

        prices_df = data.copy()
        returns_df = data.copy().pct_change()

        prices_df = prices_df[(prices_df.index.year >= start_year) & (prices_df.index.year <= end_year)]
        returns_df = returns_df[(returns_df.index.year >= start_year) & (returns_df.index.year <= end_year)]

        signals = []
        for ticker in prices_df.columns:
            tmp_prices = prices_df[[ticker]].dropna()
            tmp_returns = returns_df[[ticker]].dropna()

            lookback_windows = [1, 22, 22 * 3, 22 * 6, 22 * 12]
            lag_returns_df = pd.DataFrame(index=tmp_prices.index)
            for window in lookback_windows:
                if window == 1:
                    tmp_signal = (tmp_returns.shift(1).sort_index(axis=1)) / (tmp_returns.ewm(span=60, min_periods=60).std()).shift(1) * np.sqrt(252)
                else:
                    tmp_signal = ((tmp_returns.rolling(window=window).mean().shift(1)) / (tmp_returns.ewm(span=60, min_periods=60).std()).shift(1)) * np.sqrt(252)
                tmp_signal.rename(columns={ticker: f'lag{window}d'}, inplace=True)
                tmp_signal = tmp_signal.dropna()
                tmp_signal['ticker'] = ticker
                tmp_signal = tmp_signal.reset_index()
                tmp_signal = tmp_signal[['date', 'ticker'] + tmp_signal.columns.tolist()[1:-1]]
                lag_returns_df = pd.merge(lag_returns_df, tmp_signal, on=['date', 'ticker'], how='outer') if not lag_returns_df.empty else tmp_signal

            short_scales = [8, 16, 32]
            long_scales = [24, 48, 96]
            macd_signals_df = pd.DataFrame(index=prices_df.index)
            for S in short_scales:
                for L in long_scales:
                    if S < L:
                        ema_short = tmp_prices.ewm(span=S, adjust=False).mean()
                        ema_long = tmp_prices.ewm(span=L, adjust=False).mean()
                        macd = ema_short - ema_long
                        rolling_std = tmp_prices.rolling(window=63).std()
                        tmp_signal = (macd / (rolling_std + 1e-6)).sort_index(axis=1)
                        tmp_signal.rename(columns={ticker: f'macd_{S}_{L}'}, inplace=True)
                        tmp_signal = tmp_signal.dropna()
                        tmp_signal['ticker'] = ticker
                        tmp_signal = tmp_signal.reset_index()
                        tmp_signal = tmp_signal[['date', 'ticker'] + tmp_signal.columns.tolist()[1:-1]]
                        macd_signals_df = pd.merge(macd_signals_df, tmp_signal, on=['date', 'ticker'], how='outer') if not macd_signals_df.empty else tmp_signal

            partial_signals_df = pd.merge(lag_returns_df, macd_signals_df, on=['date', 'ticker'], how='outer')
            partial_signals_df = partial_signals_df.dropna()
            signals.append(partial_signals_df)
        signals_df = pd.concat(signals, axis=0)

        if categorical_tickers:
            signals_df['ticker_id'] = self.ticker_encoder.fit_transform(signals_df['ticker'])

        prices_df = prices_df.reset_index().melt('date').rename(columns={'value': 'price'})
        returns_tp1_df = returns_df.copy().shift(-1).reset_index().melt('date').rename(columns={'value': 'return_tp1'})
        returns_df = returns_df.copy().reset_index().melt('date').rename(columns={'value': 'return'})

        data = pd.merge(prices_df, returns_df, on=['date', 'ticker'], how='outer')
        data = pd.merge(data, returns_tp1_df, on=['date', 'ticker'], how='outer')
        data = pd.merge(data, signals_df, on=['date', 'ticker'], how='outer')

        data = data.dropna()
        data = data.sort_values(['ticker', 'date'])

        self.dt_index = signals_df['date'].sort_values().unique()

        self.num_tickers = len(signals_df['ticker'].unique())

        return data
    
    def macro_regimes_data(self, resample_freq='MS'):
        regime_output = load_pickle(os.path.join(self.inputs_path, 'regimes', 'kmeans+_manual_3_elbow.pkl'))
        regimes = regime_output['regimes']

        # filter dates
        regimes.index = pd.to_datetime(regimes.index)
        regimes = regimes.resample('MS').last()
        
        if resample_freq != 'MS':
            regimes = regimes.resample(resample_freq).last().apply(lambda x: DatasetLoader.forward_fill_until_last(x))
        
        filterd_regimes = regimes[self.dt_index[0]:]

        return filterd_regimes

    def macro_regimes_expanded_data(self, resample_freq='MS'):
        regime_output = load_pickle(os.path.join(self.inputs_path, 'regimes', 'kmeans+_manual_3_elbow.pkl'))
        
        # regime time series labels
        regimes = regime_output['regimes']
        
        # regime transition matrix
        transition_matrix = DatasetLoader.compute_transition_matrix(regimes)

        ## compute one-hot encoding of each column of regimes
        min_regime_label = int(regimes.min().min())
        max_regime_label = int(regimes.max().max())
        regime_names = [f'regime_{regime_num}' for regime_num in range(min_regime_label, max_regime_label + 1)]
        regimes_dict = {}
        for cluster_step in list(regimes.columns):
            tmp_filterd_regimes = regimes[[cluster_step]].copy().dropna().astype(int)

            if len(tmp_filterd_regimes) > 0:
                tmp_filterd_regimes = pd.get_dummies(tmp_filterd_regimes[cluster_step], prefix='regime').astype(int)

                # if there are regimes with no data, add them
                for regime_num in range(min_regime_label, max_regime_label + 1):
                    if f'regime_{regime_num}' not in tmp_filterd_regimes.columns:
                        tmp_filterd_regimes[f'regime_{regime_num}'] = 0

                # sort columns
                tmp_filterd_regimes = tmp_filterd_regimes[regime_names]
                date_label = tmp_filterd_regimes.iloc[-1].name.strftime("%Y-%m-%d")
                regimes_dict[date_label] = tmp_filterd_regimes

        # build regime probability dataframe
        transition_matrix_dict = {}
        for cluster_step in list(regimes.columns):
            tmp_filterd_regimes = regimes[[cluster_step]].copy().dropna()

            if len(tmp_filterd_regimes) > 0:
                date_label = tmp_filterd_regimes.iloc[-1].name.strftime("%Y-%m-%d")

                tmp_transition_matrix = pd.DataFrame(transition_matrix[date_label], index=regime_names, columns=regime_names)
                
                # Build regime_prob_dict using transition matrix rows
                regime_prob_dict = {
                    float(i): tmp_transition_matrix.loc[f'regime_{int(i)}'].values
                    for i in tmp_filterd_regimes[cluster_step]
                }
                regime_prob_df = pd.DataFrame.from_dict(regime_prob_dict)
                regime_prob_df.columns = [f'prob_regime_{int(i)}' for i in regime_prob_df.columns]
                regime_prob_df = regime_prob_df.reset_index()
                regime_prob_df.rename(columns={'index': cluster_step}, inplace=True)

                # merge with tmp_filterd_regimes
                features_df = pd.merge(tmp_filterd_regimes, regime_prob_df, on=cluster_step, how='left')
                features_df.index = tmp_filterd_regimes.index

                transition_matrix_dict[date_label] = features_df.drop([cluster_step], axis=1)

        # build dict with all features
        regime_features = {}
        for date_label in regimes_dict.keys():
            if (date_label in transition_matrix_dict.keys()) and (date_label in regimes_dict.keys()):
                tmp_features = pd.concat([regimes_dict[date_label], transition_matrix_dict[date_label]], axis=1)
                regime_features[date_label] = tmp_features    

        # resample date keys
        date_keys = pd.DataFrame(regime_features.keys(), columns=['date'])
        date_keys['date'] = pd.to_datetime(date_keys['date'])
        date_keys['dummy'] = 1
        date_keys = date_keys.set_index('date')
        date_keys = date_keys.resample('MS').last()
        date_keys = date_keys.resample(resample_freq).last().fillna(0)
        date_keys['day'] = date_keys.index.day
        date_keys['dummy'] = np.where((date_keys['dummy'] == 1)&(date_keys['day'] != 1), 0, date_keys['dummy'])
        del date_keys['day']

        # init list of last dates
        last_dates = []

        # first iter
        date_label = date_keys.loc[date_keys['dummy'] == 1].iloc[0].name.strftime("%Y-%m-%d")
        last_dates.append(date_label)
        resampled_regime_features = {}
        resampled_regime_features[date_label] = regime_features[date_label]
        date_keys = date_keys.drop(date_label).copy()

        for idx, row in tqdm(date_keys.iterrows(), total=date_keys.shape[0], desc="Resampling regime features..."):
            date_label = idx.strftime("%Y-%m-%d")

            # get last available date
            if row['dummy'] == 1:
                tmp_features = regime_features[date_label]
            else:
                tmp_features = resampled_regime_features[last_dates[-1]]

            # add row with nans for date_label
            row_to_add = pd.DataFrame([tmp_features.iloc[-1].values], index=[idx], columns=tmp_features.columns)
            tmp_features = pd.concat([tmp_features, row_to_add], axis=0)

            # resample features
            tmp_resampled_features = tmp_features.resample(resample_freq).last().apply(lambda x: DatasetLoader.forward_fill_until_last(x))

            # save resampled regime features
            resampled_regime_features[date_label] = tmp_resampled_features

            # save last date
            last_dates.append(idx.strftime("%Y-%m-%d"))

        return resampled_regime_features
    
    def macro_regimes_embedd_data(self, resample_freq='MS'):
        regime_output = load_pickle(os.path.join(self.inputs_path, 'regimes', 'kmeans+_manual_3_elbow.pkl'))
        
        # regime time series labels
        regimes = regime_output['regimes']
        
        # regime transition matrix
        transition_matrix = DatasetLoader.compute_transition_matrix(regimes)

        ## compute one-hot encoding of each column of regimes
        min_regime_label = int(regimes.min().min())
        max_regime_label = int(regimes.max().max())
        regime_names = [f'regime_{regime_num}' for regime_num in range(min_regime_label, max_regime_label + 1)]
        regimes_dict = {}
        for cluster_step in list(regimes.columns):
            tmp_filterd_regimes = regimes[[cluster_step]].copy().dropna().astype(int)

            if len(tmp_filterd_regimes) > 0:
                tmp_filterd_regimes[cluster_step] = self.ticker_encoder.fit_transform(tmp_filterd_regimes[cluster_step])

                # sort columns
                date_label = tmp_filterd_regimes.iloc[-1].name.strftime("%Y-%m-%d")
                tmp = []
                for lag in range(0, 6+1):
                    tmp.append(tmp_filterd_regimes.shift(lag).rename(columns={cluster_step: f'{cluster_step}_lag{lag}'}))
                tmp_df = pd.concat(tmp, axis=1).dropna()
                regimes_dict[date_label] = tmp_df.dropna()

        # build regime probability dataframe
        transition_matrix_dict = {}
        for cluster_step in list(regimes.columns):
            tmp_filterd_regimes = regimes[[cluster_step]].copy().dropna()

            if len(tmp_filterd_regimes) > 0:
                date_label = tmp_filterd_regimes.iloc[-1].name.strftime("%Y-%m-%d")

                tmp_transition_matrix = pd.DataFrame(transition_matrix[date_label], index=regime_names, columns=regime_names)
                
                # Build regime_prob_dict using transition matrix rows
                regime_prob_dict = {
                    float(i): tmp_transition_matrix.loc[f'regime_{int(i)}'].values
                    for i in tmp_filterd_regimes[cluster_step]
                }
                regime_prob_df = pd.DataFrame.from_dict(regime_prob_dict)
                regime_prob_df.columns = [f'prob_regime_{int(i)}' for i in regime_prob_df.columns]
                regime_prob_df = regime_prob_df.reset_index()
                regime_prob_df.rename(columns={'index': cluster_step}, inplace=True)

                # merge with tmp_filterd_regimes
                features_df = pd.merge(tmp_filterd_regimes, regime_prob_df, on=cluster_step, how='left')
                features_df.index = tmp_filterd_regimes.index
                features_df = features_df.drop([cluster_step], axis=1)

                # add lags
                tmp = []
                for lag in range(0, 6+1):
                    tmp_lags = features_df.copy().shift(lag)
                    tmp_lags.columns = [f'{col}_lag{lag}' for col in features_df.columns]
                    tmp.append(tmp_lags)
                features_df = pd.concat(tmp, axis=1).dropna()

                transition_matrix_dict[date_label] = features_df

        # build dict with all features
        regime_features = {}
        for date_label in regimes_dict.keys():
            if (date_label in transition_matrix_dict.keys()) and (date_label in regimes_dict.keys()):
                tmp_features = pd.concat([regimes_dict[date_label], transition_matrix_dict[date_label]], axis=1)
                regime_features[date_label] = tmp_features    

        # resample date keys
        date_keys = pd.DataFrame(regime_features.keys(), columns=['date'])
        date_keys['date'] = pd.to_datetime(date_keys['date'])
        date_keys['dummy'] = 1
        date_keys = date_keys.set_index('date')
        date_keys = date_keys.resample('MS').last()
        date_keys = date_keys.resample(resample_freq).last().fillna(0)
        date_keys['day'] = date_keys.index.day
        date_keys['dummy'] = np.where((date_keys['dummy'] == 1)&(date_keys['day'] != 1), 0, date_keys['dummy'])
        del date_keys['day']

        # init list of last dates
        last_dates = []

        # first iter
        date_label = date_keys.loc[date_keys['dummy'] == 1].iloc[0].name.strftime("%Y-%m-%d")
        last_dates.append(date_label)
        resampled_regime_features = {}
        resampled_regime_features[date_label] = regime_features[date_label]
        date_keys = date_keys.drop(date_label).copy()

        for idx, row in tqdm(date_keys.iterrows(), total=date_keys.shape[0], desc="Resampling regime features..."):
            date_label = idx.strftime("%Y-%m-%d")

            # get last available date
            if row['dummy'] == 1:
                tmp_features = regime_features[date_label]
            else:
                tmp_features = resampled_regime_features[last_dates[-1]]

            # add row with nans for date_label
            row_to_add = pd.DataFrame([tmp_features.iloc[-1].values], index=[idx], columns=tmp_features.columns)
            tmp_features = pd.concat([tmp_features, row_to_add], axis=0)

            # resample features
            tmp_resampled_features = tmp_features.resample(resample_freq).last().apply(lambda x: DatasetLoader.forward_fill_until_last(x))

            # save resampled regime features
            resampled_regime_features[date_label] = tmp_resampled_features

            # save last date
            last_dates.append(idx.strftime("%Y-%m-%d"))

        return resampled_regime_features
        
    @staticmethod
    def append_futures_expiration_dates(path):
        # list files in the directory
        files = os.listdir(path)
        # filter for .csv files
        csv_files = [f for f in files if (f.endswith('.csv')) or (f.endswith('.CSV'))]
        # loop through each file
        df_list = []
        for file in csv_files:

            ticker = path.split("/")[-1]

            # read the csv file
            df = pd.read_csv(os.path.join(path, file), header=None, usecols=range(7))
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest']
            # convert date column to datetime
            df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
            # add expiration date column
            expiry_contract = file.split('.')[0][-1]
            df['expiry_contract'] = file.split('.')[0][-1]
            # add year column
            df['year'] = file.split('.')[0].split(ticker)[-1].replace(expiry_contract, '')
            # melt the dataframe to long format
            tmp_stack_df = df.melt(['date', 'expiry_contract', 'year'])
            # append to list
            df_list.append(tmp_stack_df)
        # concatenate all dataframes
        stack_df = pd.concat(df_list, axis=0).reset_index(drop=True)

        return stack_df

    @staticmethod
    def compute_raw_carry(df, price_col='close', threshold=1000):
        """
        Computes raw carry from the two most liquid contracts per (date, year).
        Carry = (F1 - F2) / delta_T, assuming delta_T = 0.25 years.
        """

        df = df.copy()
        df['contract_id'] = df['expiry_contract'] + df['year'].astype(str)

        # Map to maturity date
        df['month'] = df['expiry_contract'].map(month_map)
        df['maturity'] = pd.to_datetime(dict(year=df['year'], month=df['month'], day=1))

        # Separate and merge
        price_df = df[df['variable'] == price_col]
        oi_df = df[df['variable'] == 'open_interest']

        merged = price_df.merge(
            oi_df[['date', 'contract_id', 'value']],
            on=['date', 'contract_id'],
            suffixes=('_price', '_oi')
        )

        # Add maturity to merged
        merged['maturity'] = price_df.set_index(['date', 'contract_id']).loc[
            merged.set_index(['date', 'contract_id']).index, 'maturity'
        ].values

        carry_list = []

        for date, group in merged.groupby('date'):
            group = group[group['value_oi'] > threshold]
            if group.shape[0] < 2:
                continue

            # Sort by expiry (i.e., maturity date)
            top2 = group.sort_values(['maturity']).iloc[:2]
            front_ct = top2.iloc[0]
            next_ct = top2.iloc[1]

            # Compute delta_T in years
            delta_t = (next_ct['maturity'] - front_ct['maturity']).days / 365.0
            if delta_t <= 0:
                continue

            carry = 1 / delta_t * (front_ct['value_price'] / next_ct['value_price'] - 1) 
            carry_list.append({'date': date, 'raw_carry': carry})

        return pd.DataFrame(carry_list).set_index('date')

    @staticmethod
    def compute_adj_carry(df, price_col='close', threshold=1000):
        raw_df = DatasetLoader.compute_raw_carry(df, price_col=price_col, threshold=threshold)

        # Compute 1Y (252-day) moving average of raw carry
        raw_df['ma_raw_carry'] = raw_df['raw_carry'].rolling(252, min_periods=20).mean()

        # Compute difference
        raw_df['carry_minus_ma'] = raw_df['raw_carry'] - raw_df['ma_raw_carry']

        # Add month info
        raw_df = raw_df.copy()
        raw_df['month'] = raw_df.index.month

        # For each month m, compute the average difference between carry and 1Y MA
        monthly_avg_adj = (
            raw_df.groupby('month')['carry_minus_ma']
            .expanding()
            .mean()
            .reset_index(level=0, drop=True)
        )

        # Merge average adjustment back
        raw_df['adj'] = monthly_avg_adj

        # Final adjusted carry
        raw_df['adj_carry'] = raw_df['raw_carry'] - raw_df['adj']

        return raw_df[['adj_carry']]

    @staticmethod
    def compute_commodity_carry(df, price_col='close', threshold=1000):
        """
        Computes commodity-specific raw carry:
        Carry_t = (F_{T+12} - F_T) / F_{T+12}
        using the front contract and the one expiring one year later.
        """
        df = df.copy()
        df['contract_id'] = df['expiry_contract'] + df['year'].astype(str)
        df['month'] = df['expiry_contract'].map(month_map)
        df['maturity'] = pd.to_datetime(dict(year=df['year'].astype(int), month=df['month'], day=1))

        # Filter for price and open interest
        price_df = df[df['variable'] == price_col]
        oi_df = df[df['variable'] == 'open_interest']

        merged = price_df.merge(
            oi_df[['date', 'contract_id', 'value']],
            on=['date', 'contract_id'],
            suffixes=('_price', '_oi')
        )

        # Add maturity date to merged dataframe
        merged['maturity'] = price_df.set_index(['date', 'contract_id']).loc[
            merged.set_index(['date', 'contract_id']).index, 'maturity'
        ].values

        carry_list = []

        for date, group in merged.groupby('date'):
            group = group[group['value_oi'] > threshold]
            if group.empty:
                continue

            # Sort by maturity
            sorted_group = group.sort_values('maturity')

            for i, front_row in sorted_group.iterrows():
                # Try to find a maturity approx. 1Y later
                target_maturity = front_row['maturity'] + pd.DateOffset(years=1)
                # Look for contract within +/- 1 month of the target
                match = sorted_group[
                    (sorted_group['maturity'] >= target_maturity - pd.DateOffset(days=30)) &
                    (sorted_group['maturity'] <= target_maturity + pd.DateOffset(days=30))
                ]

                if not match.empty:
                    back_row = match.iloc[0]
                    carry = (back_row['value_price'] - front_row['value_price']) / back_row['value_price']
                    carry_list.append({'date': date, 'adj_carry': carry})
                    break  # only one valid pair per date

        return pd.DataFrame(carry_list).set_index('date')

    @staticmethod
    def compute_carry_to_vol(df, return_col='return_tp1', span=60):
        """
        Computes raw carry to volatility ratio.
        """

        df = df.copy()
        
        # Step 1: Sort properly
        df = df.sort_values(['ticker', 'date'])

        # Step 2: Create 2-day lagged return
        df['return_lag2'] = df.groupby('ticker')[return_col].shift(2)

        # Step 3: Compute EWM std per ticker
        df['vol_ewm30'] = df.groupby('ticker')['return_lag2'].transform(lambda x: x.ewm(span=30, min_periods=30).std())
        df['vol_ewm60'] = df.groupby('ticker')['return_lag2'].transform(lambda x: x.ewm(span=span, min_periods=span).std())
        df['vol_ewm90'] = df.groupby('ticker')['return_lag2'].transform(lambda x: x.ewm(span=90, min_periods=90).std())

        # rolling skewness
        df['skew_ewm30'] = df.groupby('ticker')['return_lag2'].transform(lambda x: x.rolling(window=30, min_periods=30).skew())
        df['skew_ewm60'] = df.groupby('ticker')['return_lag2'].transform(lambda x: x.rolling(window=span, min_periods=span).skew())
        df['skew_ewm90'] = df.groupby('ticker')['return_lag2'].transform(lambda x: x.rolling(window=90, min_periods=90).skew())

        # rolling kurtosis
        df['kurt_ewm30'] = df.groupby('ticker')['return_lag2'].transform(lambda x: x.rolling(window=30, min_periods=30).kurt())
        df['kurt_ewm60'] = df.groupby('ticker')['return_lag2'].transform(lambda x: x.rolling(window=span, min_periods=span).kurt())
        df['kurt_ewm90'] = df.groupby('ticker')['return_lag2'].transform(lambda x: x.rolling(window=90, min_periods=90).kurt())

        # vol diff
        df['vol_ewm30_diff'] = df.groupby('ticker')['vol_ewm30'].diff(1)
        df['vol_ewm60_diff'] = df.groupby('ticker')['vol_ewm60'].diff(1)
        df['vol_ewm90_diff'] = df.groupby('ticker')['vol_ewm90'].diff(1)

        # Step 4: Compute ratios
        df['raw_carry_to_vol'] = df['raw_carry'] / df['vol_ewm60']
        df['adj_carry_to_vol'] = df['adj_carry'] / df['vol_ewm60']

        # carry diff
        df['raw_carry_diff'] = df.groupby('ticker')['raw_carry'].diff(1)
        df['adj_carry_diff'] = df.groupby('ticker')['adj_carry'].diff(1)
        df['raw_carry_to_vol_diff'] = df.groupby('ticker')['raw_carry_to_vol'].diff(1)
        df['adj_carry_to_vol_diff'] = df.groupby('ticker')['adj_carry_to_vol'].diff(1)

        # drop unnecessary columns
        # df = df.drop(columns=['return_lag2', 'vol_ewm60'], axis=1)
        df = df.drop(columns=['return_lag2'], axis=1)

        return df.dropna().reset_index(drop=True)
    
    def carry_data(self,
                   dataset_name,
                   continuous_future_method,
                   start_year,
                   end_year,
                   load_data=True):
        # build features (signals) dataset
        if not load_data:
            carry_list = []
            tickers_unavailable = []
            for asset_class, tickers in tqdm(asset_classes.items(), desc='Computing carry for all asset classes', total=len(asset_classes)):
                for ticker in tickers:

                    # check if the ticker is available
                    if not os.path.exists(os.path.join(self.inputs_path, 'pinnacle', 'INDIVIDUAL_COMMODITY_CONTRACTS', ticker)):
                        tickers_unavailable.append(ticker)
                        continue

                    # append futures expiration dates
                    stack_df = DatasetLoader.append_futures_expiration_dates(os.path.join(self.inputs_path, 'pinnacle', 'INDIVIDUAL_COMMODITY_CONTRACTS', ticker))

                    # compute raw carry
                    raw_carry_df = DatasetLoader.compute_raw_carry(stack_df, price_col='close', threshold=100)

                    # compute adjusted carry
                    if asset_class == 'commodities':
                        try:
                            adj_carry_df = DatasetLoader.compute_commodity_carry(stack_df, price_col='close', threshold=100)
                        except:
                            adj_carry_df = DatasetLoader.compute_commodity_carry(stack_df, price_col='close', threshold=100)
                            raise Exception(f"Error computing adjusted carry for {ticker}")
                    else:
                        adj_carry_df = DatasetLoader.compute_adj_carry(stack_df, price_col='close', threshold=100)

                    # stack carry measures
                    new_ticker = ticker.replace('_', '')
                    stack_raw_carry_df = raw_carry_df.reset_index().melt('date')
                    stack_raw_carry_df['ticker'] = new_ticker
                    stack_raw_carry_df['asset_class'] = asset_class

                    stack_adj_carry_df = adj_carry_df.reset_index().melt('date')
                    stack_adj_carry_df['ticker'] = new_ticker
                    stack_adj_carry_df['asset_class'] = asset_class

                    # concatenate all dataframes
                    tmp_stack_carry_df = pd.concat([stack_raw_carry_df, stack_adj_carry_df], axis=0)

                    # filter dates
                    mask = (tmp_stack_carry_df['date'].dt.year >= start_year)&(tmp_stack_carry_df['date'].dt.year <= end_year)
                    tmp_stack_carry_df = tmp_stack_carry_df[mask]

                    carry_list.append(tmp_stack_carry_df)
            carry_df = pd.concat(carry_list, axis=0)

            # save carry features (signals)
            carry_df.to_csv(os.path.join(self.inputs_path, 'pinnacle', 'processed', 'carry_signals.csv'), index=False)
        else:
            carry_df = pd.read_csv(os.path.join(self.inputs_path, 'pinnacle', 'processed', 'carry_signals.csv'))
            carry_df['date'] = pd.to_datetime(carry_df['date'], format='%Y-%m-%d')

        # refactor columns
        selected_index = ['date', 'ticker', 'asset_class']
        carry_df = carry_df.pivot_table(index=selected_index, columns=['variable'], values=['value'])
        carry_df.columns = carry_df.columns.droplevel()
        carry_df = carry_df.reset_index()

        # build target dataset
        data = self.build_futures_summary_data(continuous_future_method)
        data = self.generic_melted_loader(dataset_name, data)
        data = data.resample('B').last().ffill()
        lagged_returns_df = data.copy().pct_change(1).shift(-1)

        # stack target
        stack_lagged_returns_df = lagged_returns_df.reset_index().melt('date')
        stack_lagged_returns_df = stack_lagged_returns_df.rename(columns={'value': 'return_tp1'})

        # merge target with features
        signals_df = pd.merge(stack_lagged_returns_df, carry_df, on=['date', 'ticker'], how='left').dropna()

        # compute carry to vol
        signals_df = DatasetLoader.compute_carry_to_vol(signals_df, return_col='return_tp1', span=60)

        # stantardize all features
        FEATURE_NAMES = list(signals_df.drop(selected_index + ['return_tp1'], axis=1).columns)
        for fn in FEATURE_NAMES:
            signals_df[f'{fn}_rolling_mean'] = signals_df.groupby('ticker')[fn].transform(lambda x: x.rolling(60).mean())
            signals_df[f'{fn}_rolling_std'] = signals_df.groupby('ticker')[fn].transform(lambda x: x.rolling(60).std())
            signals_df[fn] = (signals_df[fn] - signals_df[f'{fn}_rolling_mean']) / signals_df[f'{fn}_rolling_std']

            signals_df = signals_df.drop([f'{fn}_rolling_mean', f'{fn}_rolling_std'], axis=1)
        signals_df = signals_df.dropna()

        
        signals_df['ticker_id'] = self.ticker_encoder.fit_transform(signals_df['ticker'])
        signals_df['asset_class_id'] = self.ticker_encoder.fit_transform(signals_df['asset_class'])

        self.dt_index = signals_df['date'].sort_values().unique()

        self.num_tickers = len(signals_df['ticker'].unique())
        self.num_asset_classes = len(signals_df['asset_class'].unique())

        del signals_df['asset_class']

        return signals_df

    def carry_data2(self,
                   dataset_name,
                   continuous_future_method,
                   start_year,
                   end_year,
                   categorical_tickers=False,
                   load_data=True):
        tickers = self.flds.keys()
        out = []
        for ticker in tickers:
            if len(ticker) < 2:
                ticker = f"{ticker}_"

            path = os.path.join(self.inputs_path, 'pinnacle', 'INDIVIDUAL_COMMODITY_CONTRACTS', ticker)
            individual_futures_df = DatasetLoader.append_futures_expiration_dates(path)

            close_stack_df = individual_futures_df.loc[individual_futures_df['variable'] == 'close'].copy()

            # add fake maturity column
            close_stack_df['month'] = close_stack_df['expiry_contract'].map(month_map)
            close_stack_df['maturity'] = pd.to_datetime(dict(year=close_stack_df['year'], month=close_stack_df['month'], day=1))

            # add next fake maturity and value columns
            close_stack_df = close_stack_df.sort_values(['date', 'maturity']).reset_index(drop=True)
            close_stack_df['next_maturity'] = close_stack_df.groupby(['date'])['maturity'].shift(-1)
            close_stack_df['next_value'] = close_stack_df.groupby(['date'])['value'].shift(-1)
            close_stack_df = close_stack_df.dropna(subset=['next_maturity']).copy()

            # # add next value column
            # close_stack_df = close_stack_df.sort_values(['date', 'maturity']).reset_index(drop=True)
            # value_map = close_stack_df[['date', 'maturity', 'value']].copy()
            # value_map = value_map.rename(columns={'maturity': 'next_maturity', 'value': 'next_value'})
            # close_stack_df = close_stack_df.merge(value_map, on=['date', 'next_maturity'], how='left')

            # compute time difference in days between the current and next maturity
            close_stack_df['delta_T'] = (close_stack_df['next_maturity'] - close_stack_df['maturity']).dt.days

            # compute raw carry
            close_stack_df['raw_carry'] = (1/close_stack_df['delta_T']) * (close_stack_df['value'] - close_stack_df['next_value'])

            # create a dataframe with date and raw_carry
            carry_df = close_stack_df.sort_values(['date', 'maturity'])[['date', 'raw_carry']].copy()
            carry_df = carry_df.set_index(['date'])
            carry_df = carry_df.resample('B').first().ffill().copy()

            # compute adj carry = 1/n \sum_{i=1}^{n} raw_carry_i - ma(raw_carry_{1:i}, n)
            carry_df['adj'] = carry_df['raw_carry'] - carry_df['raw_carry'].rolling(window=252, min_periods=30).mean()
            carry_df['adj'] = carry_df['adj'].rolling(window=carry_df.shape[0], min_periods=30).mean()
            carry_df['carry'] = carry_df['raw_carry'] - carry_df['adj']

            # standardize carry
            carry_df['carry_zscore'] = (carry_df['carry'] - carry_df['carry'].rolling(window=90, min_periods=90).mean().shift(+1)) / carry_df['carry'].rolling(window=90, min_periods=90).std().shift(+1)

            carry_df['ticker'] = ticker
            melt_carry_df = carry_df.reset_index()[['date', 'ticker', 'carry', 'carry_zscore']].copy().melt(['date', 'ticker']).pivot_table(index=['date', 'ticker'], columns='variable', values='value').reset_index()

            file_name = f"{ticker}_{continuous_future_method}.CSV"
            tmp_data = pd.read_csv(os.path.join(self.inputs_path, 'pinnacle', 'CLCDATA', file_name))
            ticker = file_name.split('_')[0]
            tmp_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'open int']
            tmp_data['date'] = pd.to_datetime(tmp_data['date'], format='%m/%d/%Y')
            tmp_data = tmp_data[['date', 'close']]
            tmp_data['return'] = tmp_data['close'].pct_change()
            tmp_data['volatility'] = tmp_data['return'].rolling(window=90, min_periods=30).std().shift(1) * np.sqrt(252)
            tmp_data['return_tp1'] = tmp_data['return'].shift(-1)
            tmp_data['ticker'] = ticker
            tmp_data = tmp_data.dropna()[['date', 'ticker', 'return_tp1', 'volatility']].copy()
         
            melt_returns_df = tmp_data.melt(['date', 'ticker']).pivot_table(index=['date', 'ticker'], columns='variable', values='value').reset_index()

            tmp_out_df = melt_returns_df.merge(melt_carry_df, on=['date', 'ticker'], how='left').dropna().reset_index(drop=True)

            # carry to vol
            tmp_out_df['carry_to_vol'] = tmp_out_df['carry'] / tmp_out_df['volatility']

            # carry to vol zscore
            tmp_out_df['carry_to_vol_zscore_90'] = (tmp_out_df['carry_to_vol'] - tmp_out_df['carry_to_vol'].rolling(window=90, min_periods=90).mean().shift(+1)) / tmp_out_df['carry_to_vol'].rolling(window=90, min_periods=90).std().shift(+1)
            tmp_out_df['carry_to_vol_zscore_180'] = (tmp_out_df['carry_to_vol'] - tmp_out_df['carry_to_vol'].rolling(window=180, min_periods=90).mean().shift(+1)) / tmp_out_df['carry_to_vol'].rolling(window=180, min_periods=90).std().shift(+1)
            tmp_out_df['carry_to_vol_zscore_270'] = (tmp_out_df['carry_to_vol'] - tmp_out_df['carry_to_vol'].rolling(window=270, min_periods=90).mean().shift(+1)) / tmp_out_df['carry_to_vol'].rolling(window=270, min_periods=90).std().shift(+1)
            tmp_out_df['carry_to_vol_zscore_360'] = (tmp_out_df['carry_to_vol'] - tmp_out_df['carry_to_vol'].rolling(window=360, min_periods=90).mean().shift(+1)) / tmp_out_df['carry_to_vol'].rolling(window=360, min_periods=90).std().shift(+1)

            # drop volatility column and nan rows
            tmp_out_df = tmp_out_df.drop(columns=['volatility'], axis=1).dropna()

            # add carry sign as a feature
            tmp_out_df['carry_sign'] = np.where(tmp_out_df['carry'] > 0, 1, -1)

            # add numerical binning of the carry signal into five bins
            tmp_out_df['carry_bin'] = pd.cut(tmp_out_df['carry'], bins=[-np.inf, -0.01, 0.01, 0.02, 0.03, np.inf], labels=[0, 1, 2, 3, 4])

            # rolling window param
            window = 252  # or ano
            
            def assign_rolling_bin(row):
                carry_val = row['carry']
                q20, q40, q60, q80 = row['q20'], row['q40'], row['q60'], row['q80']
                
                if pd.isna(q20):  # Handle cases where rolling quantiles aren't available yet
                    return np.nan
                elif carry_val <= q20:
                    return 0
                elif carry_val <= q40:
                    return 1
                elif carry_val <= q60:
                    return 2
                elif carry_val <= q80:
                    return 3
                else:
                    return 4

            # calculate rolling quantiles (20th, 40th, 60th, 80th percentiles)
            # assign quantiles to dataframe and apply binning
            tmp_out_df['q20'] = tmp_out_df['carry'].rolling(window=window, min_periods=30).quantile(0.2)
            tmp_out_df['q40'] = tmp_out_df['carry'].rolling(window=window, min_periods=30).quantile(0.4)
            tmp_out_df['q60'] = tmp_out_df['carry'].rolling(window=window, min_periods=30).quantile(0.6)
            tmp_out_df['q80'] = tmp_out_df['carry'].rolling(window=window, min_periods=30).quantile(0.8)

            tmp_out_df['carry_bin'] = tmp_out_df.apply(assign_rolling_bin, axis=1)

            tmp_out_df['carry_q01'] = tmp_out_df['carry'].rolling(window=window, min_periods=window).quantile(0.01)
            tmp_out_df['carry_q99'] = tmp_out_df['carry'].rolling(window=window, min_periods=window).quantile(0.99)
            tmp_out_df['carry_clip'] = tmp_out_df['carry'].clip(lower=tmp_out_df['carry_q01'], upper=tmp_out_df['carry_q99'])

            tmp_out_df['carry_to_vol_q01'] = tmp_out_df['carry_to_vol'].rolling(window=window, min_periods=window).quantile(0.01)
            tmp_out_df['carry_to_vol_q99'] = tmp_out_df['carry_to_vol'].rolling(window=window, min_periods=window).quantile(0.99)
            tmp_out_df['carry_to_vol_clip'] = tmp_out_df['carry_to_vol'].clip(lower=tmp_out_df['carry_to_vol_q01'], upper=tmp_out_df['carry_to_vol_q99'])

            tmp_out_df['carry_zscore_q01'] = tmp_out_df['carry_zscore'].rolling(window=window, min_periods=window).quantile(0.01)
            tmp_out_df['carry_zscore_q99'] = tmp_out_df['carry_zscore'].rolling(window=window, min_periods=window).quantile(0.99)
            tmp_out_df['carry_zscore_clip'] = tmp_out_df['carry_zscore'].clip(lower=tmp_out_df['carry_zscore_q01'], upper=tmp_out_df['carry_zscore_q99'])

            tmp_out_df['carry_to_vol_zscore_90_q01'] = tmp_out_df['carry_to_vol_zscore_90'].rolling(window=window, min_periods=window).quantile(0.01)
            tmp_out_df['carry_to_vol_zscore_90_q99'] = tmp_out_df['carry_to_vol_zscore_90'].rolling(window=window, min_periods=window).quantile(0.99)
            tmp_out_df['carry_to_vol_zscore_90_clip'] = tmp_out_df['carry_to_vol_zscore_90'].clip(lower=tmp_out_df['carry_to_vol_zscore_90_q01'], upper=tmp_out_df['carry_to_vol_zscore_90_q99'])

            tmp_out_df['carry_to_vol_zscore_180_q01'] = tmp_out_df['carry_to_vol_zscore_180'].rolling(window=window, min_periods=window).quantile(0.01)
            tmp_out_df['carry_to_vol_zscore_180_q99'] = tmp_out_df['carry_to_vol_zscore_180'].rolling(window=window, min_periods=window).quantile(0.99)
            tmp_out_df['carry_to_vol_zscore_180_clip'] = tmp_out_df['carry_to_vol_zscore_180'].clip(lower=tmp_out_df['carry_to_vol_zscore_180_q01'], upper=tmp_out_df['carry_to_vol_zscore_180_q99'])

            tmp_out_df['carry_to_vol_zscore_270_q01'] = tmp_out_df['carry_to_vol_zscore_270'].rolling(window=window, min_periods=window).quantile(0.01)
            tmp_out_df['carry_to_vol_zscore_270_q99'] = tmp_out_df['carry_to_vol_zscore_270'].rolling(window=window, min_periods=window).quantile(0.99)
            tmp_out_df['carry_to_vol_zscore_270_clip'] = tmp_out_df['carry_to_vol_zscore_270'].clip(lower=tmp_out_df['carry_to_vol_zscore_270_q01'], upper=tmp_out_df['carry_to_vol_zscore_270_q99'])
            
            tmp_out_df['carry_to_vol_zscore_360_q01'] = tmp_out_df['carry_to_vol_zscore_360'].rolling(window=window, min_periods=window).quantile(0.01)
            tmp_out_df['carry_to_vol_zscore_360_q99'] = tmp_out_df['carry_to_vol_zscore_360'].rolling(window=window, min_periods=window).quantile(0.99)
            tmp_out_df['carry_to_vol_zscore_360_clip'] = tmp_out_df['carry_to_vol_zscore_360'].clip(lower=tmp_out_df['carry_to_vol_zscore_360_q01'], upper=tmp_out_df['carry_to_vol_zscore_360_q99'])

            tmp_out_df = tmp_out_df.drop(columns=[
                'carry_q01', 'carry_q99', 
                'carry_to_vol_q01', 'carry_to_vol_q99', 
                'carry_zscore_q01', 'carry_zscore_q99', 
                'carry_to_vol_zscore_90_q01', 'carry_to_vol_zscore_90_q99', 
                'carry_to_vol_zscore_180_q01', 'carry_to_vol_zscore_180_q99',
                'carry_to_vol_zscore_270_q01', 'carry_to_vol_zscore_270_q99', 
                'carry_to_vol_zscore_360_q01', 'carry_to_vol_zscore_360_q99', 
                'q20', 'q40', 'q60', 'q80'
            ], axis=1)

            out.append(tmp_out_df.ffill().dropna())
        out_df = pd.concat(out, axis=0).reset_index(drop=True)

        # filter dates
        out_df = out_df.loc[(out_df['date'].dt.year >= start_year) & (out_df['date'].dt.year <= end_year)].copy().reset_index(drop=True)

        if categorical_tickers:
            out_df['ticker_id'] = self.ticker_encoder.fit_transform(out_df['ticker'])

        self.num_tickers = len(out_df['ticker'].unique())

        return out_df

    @staticmethod
    def compute_expanding_top_k_eigenvalues(returns_df, min_obs=20, k=3):
        """
        Compute the top-k eigenvalues from the expanding covariance matrix of returns.

        Parameters:
        - returns_df: DataFrame of returns (index=date, columns=tickers)
        - min_obs: minimum number of days before starting calculation
        - k: number of top eigenvalues to extract

        Returns:
        - DataFrame with index=date and columns=['eigval_1', ..., 'eigval_k']
        """
        top_k_eigenvalues = []

        for i in range(min_obs, len(returns_df)):
            window_returns = returns_df.iloc[:i].dropna(axis=1, how='any')  # drop assets with missing data

            cov_matrix = window_returns.cov().values
            eigvals = np.linalg.eigvalsh(cov_matrix)  # use eigvalsh for symmetric matrices

            top_k = np.sort(eigvals)[-k:][::-1]  # sort descending and take top-k
            top_k_df = pd.DataFrame(top_k).T
            top_k_df.columns = [f'eigval_{j+1}' for j in range(top_k_df.shape[1])]
            top_k_df.index = [returns_df.index[i]]
            top_k_eigenvalues.append(top_k_df)

        return pd.concat(top_k_eigenvalues)

    def futures_stock_bond_optmization_data(self,
                                            dataset_name,
                                            continuous_future_method,
                                            start_year,
                                            end_year,
                                            categorical_tickers):

        data = self.build_futures_summary_data(continuous_future_method)
        data = self.generic_melted_loader(dataset_name, data)
        data = data.resample('B').last().ffill()

        prices_df = data.copy()
        returns_df = data.copy().pct_change()

        # compute expnding covariance for each time step of returns
        k_eigenvalues_df = DatasetLoader.compute_expanding_top_k_eigenvalues(returns_df.dropna(), min_obs=20, k=3)

        prices_df = prices_df[(prices_df.index.year >= start_year) & (prices_df.index.year <= end_year)]
        returns_df = returns_df[(returns_df.index.year >= start_year) & (returns_df.index.year <= end_year)]

        # stack dataset
        stack_returns_df = returns_df.reset_index().melt('date').rename(columns={'value': 'return_lag1'})
        stack_prices_df = prices_df.reset_index().melt('date').rename(columns={'value': 'price_lag1'})

        # add return_tp1
        stack_returns_df['return_tp1'] = stack_returns_df.groupby('ticker')['return_lag1'].shift(-1)

        # merge dataframes
        out_df = pd.merge(stack_returns_df, stack_prices_df, on=['date', 'ticker'], how='left')

        # reoreder columns
        out_df = out_df[['date', 'ticker', 'return_tp1', 'return_lag1', 'price_lag1']]

        # compute rolling mean
        out_df['expanding_mean'] = out_df.groupby('ticker')['return_lag1'].transform(lambda x: x.rolling(len(x), min_periods=20).mean())

        # compute rolling std
        out_df['expanding_std'] = out_df.groupby('ticker')['return_lag1'].transform(lambda x: x.rolling(len(x), min_periods=20).std())

        # compute eigenvalues
        out_df = pd.merge(out_df, k_eigenvalues_df.reset_index().rename(columns={'index': 'date'}), on='date', how='left')        

        if categorical_tickers:
            out_df['ticker_id'] = self.ticker_encoder.fit_transform(out_df['ticker'])

        self.dt_index = out_df['date'].sort_values().unique()

        return out_df.dropna()

    def macro_regimes_raw_data(self, resample_freq='MS'):
        fredmd_transf_df = pd.read_csv(os.path.join(self.inputs_path, 'fredmd', 'fredmd_transf_df.csv'))
        fredmd_des_df = pd.read_csv(os.path.join(self.inputs_path, 'fredmd', 'fredmd_description.csv'), delimiter=";")

        # filter columns
        fred_names_transf = pd.DataFrame(fredmd_transf_df.drop(['date'], axis=1).columns.tolist(), columns=['fred'])
        merge_names = pd.merge(fred_names_transf, fredmd_des_df[['fred', 'group']], on='fred', how='left')
        selected_fred_names = merge_names.loc[~merge_names['group'].isin(['Stock Market', 'Interest and Exchange Rates'])].copy()
        fredmd_transf_df = fredmd_transf_df[['date'] + selected_fred_names['fred'].tolist()]

        # fix dates
        fredmd_transf_df["date"] = pd.to_datetime(fredmd_transf_df["date"])
        fredmd_transf_df = fredmd_transf_df.set_index("date")

        # fix formatting
        fredmd_transf_df = fredmd_transf_df.astype(float)

        # lag all features by 1 month
        fredmd_transf_df.index = fredmd_transf_df.index + pd.DateOffset(months=1)

        # rolling zscore
        zscore_fredmd_transf_df = fredmd_transf_df.copy()
        for col in zscore_fredmd_transf_df.columns:
            tmp_rolling_mean = zscore_fredmd_transf_df[col].rolling(12, min_periods=3).mean().shift(+1)
            tmp_rolling_std = zscore_fredmd_transf_df[col].rolling(12, min_periods=3).std().shift(+1)
            tmp_rolling_zscore = (zscore_fredmd_transf_df[col] - tmp_rolling_mean) / tmp_rolling_std
            zscore_fredmd_transf_df[col] = tmp_rolling_zscore

        zscore_fredmd_transf_df = zscore_fredmd_transf_df.dropna()

        # resample data
        if resample_freq is not None:
            zscore_fredmd_transf_df = zscore_fredmd_transf_df.resample(resample_freq).last().ffill().dropna()

        return zscore_fredmd_transf_df
    
    def futures_data_only(self, dataset_name, continuous_future_method):
        data = self.build_futures_summary_data(continuous_future_method)
        data = self.generic_melted_loader(dataset_name, data)
        data = data.resample('B').last().ffill()
        data = data.dropna()

        return data

    def load_data(self, dataset_name: str, **kwargs):
        if dataset_name == 'tsmom':
            data = self.lzr_tsmom_data(
                dataset_name,
                kwargs['continuous_future_method'],
                kwargs['start_year'],
                kwargs['end_year'],
                kwargs['categorical_tickers'],
            )
        elif dataset_name == 'futures':
            data = self.futures_data_only(dataset_name, kwargs['continuous_future_method'])
        else:
            raise Exception('Dataset not found {dataset_name}')

        return data

if __name__ == '__main__':
    if DEBUG:
        dataset_name = 'futures'
        continuous_future_method = 'RAD'

        ds_builder = DatasetLoader(
            flds={

            # commodities
            'CC': ['close'], 'DA': ['close'], 'GI': ['close'], 'JO': ['close'], 'KC': ['close'], 'KW': ['close'],
            'LB': ['close'], 'NR': ['close'], 'SB': ['close'], 'ZC': ['close'], 'ZF': ['close'], 'ZZ': ['close'],
            'ZG': ['close'], 'ZH': ['close'], 'ZI': ['close'], 'ZK': ['close'], 'ZL': ['close'], 'ZN': ['close'],
            'ZO': ['close'], 'ZP': ['close'], 'ZR': ['close'], 'ZT': ['close'], 'ZU': ['close'], 'ZW': ['close'],
            
            # bonds
            'CB': ['close'], 'DT': ['close'], 'EC': ['close'], 'FB': ['close'], 'GS': ['close'], 'TU': ['close'], 
            'TY': ['close'], 'UB': ['close'], 'US': ['close'], 'UZ': ['close'], 
            
            # fx
            'AN': ['close'], 'CN': ['close'], 'BN': ['close'], 'DX': ['close'], 'JN': ['close'], 'MP': ['close'], 'SN': ['close'],

            # # equities
            'FN': ['close'], 'NK': ['close'], 'ZA': ['close'], 'CA': ['close'], 'EN': ['close'], 'ER': ['close'], 'ES': ['close'],
            'LX': ['close'], 'MD': ['close'], 'SC': ['close'], 'SP': ['close'], 'XU': ['close'], 'XX': ['close'], 'YM': ['close'],
            'NK': ['close'],
        }
    )

    data = ds_builder.load_data(
        dataset_name=dataset_name,
        continuous_future_method=continuous_future_method,
    )
