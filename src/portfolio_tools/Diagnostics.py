import pandas as pd
import torch

from loss_functions.ExpectedRet import ExpectedRet
from loss_functions.Volatility import Volatility
from loss_functions.Sharpe import Sharpe
from loss_functions.Sortino import Sortino
from loss_functions.AverageDD import AverageDD
from loss_functions.MaxDD import MaxDD
from loss_functions.PositiveRetRatio import PositiveRetRatio

class Diagnostics:
    def __init__(self) -> None:
        pass

    def compute_summary_statistics(self,
                                   portfolio_returns: pd.DataFrame,
                                   default_metrics: list = [ExpectedRet, Volatility, Sharpe, Sortino, AverageDD, MaxDD, PositiveRetRatio]):
        
        torch_portfolio_returns = torch.tensor(portfolio_returns.dropna().values)

        portfolio_stats = {}
        for metric in default_metrics:
            metric = metric()
            portfolio_stats[metric.name] = metric.forward(returns=torch_portfolio_returns).item()

        return portfolio_stats