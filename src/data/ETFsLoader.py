import os
import torch
import pandas as pd
import numpy as np

class ETFsLoader(object):
    """

    
    """
    
    def __init__(self, tickers: list=None):
        super().__init__()

        self.inputs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "inputs")
        self.tickers = tickers
        self._read_data()

    def _read_data(self):
        
        etfs_df = pd.read_csv(os.path.join(self.inputs_path, "sample", "etfs.csv"), sep=";")
        etfs_df["date"] = pd.to_datetime(etfs_df["date"])
        etfs_df.set_index("date", inplace=True)

        if self.tickers is not None:
            etfs_df = etfs_df[self.tickers]

        # dataset processing 1
        ## sort index
        etfs_df = etfs_df.sort_index()

        # dataset processing 2
        ## compute returns and subset data
        prices_df = etfs_df.dropna().copy()
        returns_df = np.log(etfs_df).diff().dropna().copy()

        self.prices = prices_df
        self.returns = returns_df

