import torch
import numpy as np
import pandas as pd

class TSM:
    """
    This class implements different formulations of the time series momentum signal.

    Paper: https://arxiv.org/pdf/1904.04912.pdf
    """
    
    def __init__(self) -> None:
        pass

    def Moskowitz(self, prices: pd.DataFrame, window: int=252) -> torch.Tensor:
        """
        Moskowitz Method to compute the strategy for asset allocation.
        
        Returns:
            position_sizing (torch.tensor): Moskowitz method for Position sizing with +1 or -1
        """

        # compute log returns
        log_returns = np.log(prices / prices.shift(1))

        # obtain the rolling mean of the returns of all assets in the past L days
        window_log_returns = log_returns.rolling(window=window).mean()

        return window_log_returns
    
    # BAZ method implementation
    def Baz(self,
            L: int,
            S: int) -> torch.Tensor:
        """
        Baz Method to compute the strategy for asset allocation.
        Args:
        L: long time scale (24,48,96)
        S: short time scale (8,16,32)
        Returns:
            position_sizing (torch.tensor): Moskowitz method for Position sizing with values between +1 or -1
        """
        N = self.time_series.shape[0]
        MACD = self.EWMA(S) - self.EWMA(L) 
        # standard deviation of the last 63 days
        STD = torch.std(self.time_series_window[(N - 63):N,:],axis = 0)
        STD_year = torch.std(self.time_series_window[(N - 252):N,:],axis = 0)
        # calculate Q
        Q = MACD/STD
        #
        Y = Q/STD_year
        # compute positions
        position_sizing = (Y*torch.exp(-(Y*Y)/4.0))/0.89

        return position_sizing
    
    # Evaluate Strategy (TSOM: Time series momentum)
    def evaluate_strategy(self,
                          real_returns: torch.Tensor,
                          #predicted_returns: torch.Tensor,
                          method: str = "Moskowitz",
                          L: int = 24,
                          S: int = 8) -> torch.Tensor:
        # calculate strategy
        position_sizing = None
        if method == "Moskowitz":
            position_sizing = self.Moskowitz()
        else: # Baz method selected
            position_sizing = self.Baz(L,S)
        # VOLATILITY
        sigma_TGT = 0.15
        sigma_t = self.EWSD(60)
        # compute TSOM
        return_TSOM = torch.mean(sigma_TGT*((position_sizing*real_returns)/sigma_t))
        # return value
        return return_TSOM


    # Exponential Weighted Average of all assets of the last S days
    # TO DO: correct when S is larger than the number of time series points!
    def EWMA(self,
             S: int) -> torch.Tensor:
        N = self.time_series.shape[0]
        alpha = (S-1)/S
        time_series_window = self.time_series[(N - S):N,:]
        weights = torch.ones(1, S)
        for idx in range(S-2,-1,-1):
            weights[0,idx] = (1-alpha)*weights[0,idx + 1]
        #
        weights = alpha*weights
        #
        return (torch.matmul(weights, time_series_window))
    
    # Exponential Weighted Standard Deviation of all assets of the last S days
    # TO DO: correct when S is larger than the number of time series points!
    def EWSD(self,
             S: int) -> torch.Tensor:
        N = self.time_series.shape[0]
        alpha = (S-1)/S
        time_series_window = self.time_series[(N - S):N,:]
        weights = torch.ones(S,1)
        for idx in range(S-2,-1,-1):
            weights[idx,0] = (1-alpha)*weights[idx + 1,0]
        #
        wtime_series_window = time_series_window*weights
        mean_wtime_series_window = torch.sum(wtime_series_window,axis = 0)
        sd_wtime_series_window =  (wtime_series_window - mean_wtime_series_window)*(wtime_series_window - mean_wtime_series_window)
        #
        return (torch.mean(sd_wtime_series_window, axis = 0))