import pandas as pd
import os
from glob import glob
from tqdm import tqdm

from settings import INPUT_PATH
from utils.conn_data import save_pickle

if __name__ == '__main__':
    flds = ['date', 'ticker', 'open', 'high', 'low', 'close', 'prevAdjClose', 'prevAdjOpen', 'pvCLCL']

    # list files in directory
    years = os.listdir(os.path.join(INPUT_PATH, 'US_CRSP_NYSE'))

    out_df_list = []
    for y in tqdm(years, total=len(years), desc="Parsing CRSP files"):
        # list .csv.gz files in directory
        files = glob(os.path.join(INPUT_PATH, 'US_CRSP_NYSE', y, '*.csv.gz'))
        for f in files:
            # read .csv.gz file
            df = pd.read_csv(f, compression='gzip')
            df["date"] = pd.to_datetime(f.split("/")[-1].split(".")[0])

            df = df[flds]
            df["curAdjClose"] = (1 + df["pvCLCL"]) * df["prevAdjClose"]

            out_df_list.append(df)
    out_df = pd.concat(out_df_list, axis=0) 

    out_dict = {}
    for ticker in tqdm(out_df.ticker.unique(), total=len(out_df.ticker.unique()), desc="Creating output dictionary"):
        out_dict[ticker] = out_df.loc[out_df["ticker"] == ticker].drop(["ticker"], axis=1).sort_values("date").set_index('date')

    save_pickle(out_dict, os.path.join(INPUT_PATH, 'crsp_nyse.pickle'))