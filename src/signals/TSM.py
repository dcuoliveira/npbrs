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

    def Moskowitz(self, returns: pd.DataFrame, window: int=252) -> torch.Tensor:
        """
        Moskowitz Method to compute the strategy for asset allocation.
        
        Returns:
            position_sizing (torch.tensor): Moskowitz method for Position sizing with +1 or -1

        Paper: Moskowitz et al. (2012) - Time Series Momentum.
        """

        # obtain the rolling mean of the returns of all assets in the past L days
        momentum = returns.rolling(window=window).mean()

        return momentum
    
    def CTA(self, returns: pd.DataFrame, short_term: list, long_term: list, sw: int, pw: int, weights: list) -> torch.Tensor:
        """
        Commodity Trading Advisor (CTA) Method to compute the strategy for asset allocation.
        
        Returns:
            position_sizing (torch.tensor): CTA method for Position sizing with +1 or -1

        Paper: Baz et al. (2015) - Dissecting Investment Strategies in the Cross Sectionand Time Series.
        """

        if range(len(short_term)) == range(len(long_term)):
            ks = range(len(short_term)) 
        else:
            raise ValueError("Short and Long term should be of same length")

        intermediate_signals = []
        for k in ks:

            # compute short and long-term ewma
            short_term_signal = returns.ewm(halflife=short_term[k]).mean()
            long_term_signal = returns.ewm(halflife=long_term[k]).mean()

            # compute diff
            signal_diff = short_term_signal - long_term_signal

            # standardize diff
            standardized_diff = signal_diff / returns.rolling(window=sw).std()

            # standardize (standardized_diff)
            standardized_diff = standardized_diff / standardized_diff.rolling(window=pw).std()

            intermediate_signal = (standardized_diff * np.exp(-0.25 * (standardized_diff ** 2))) / 0.89
            intermediate_signals.append(intermediate_signal)

        for i, w in enumerate(weights):
            if i == 0:
                momentum = w * intermediate_signals[i]
            else:
                momentum += w * intermediate_signals[i]

        return momentum