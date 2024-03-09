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
        self.str_to_metric = {
            'ExpectedRet': ExpectedRet,
            'Volatility': Volatility,
            'Sharpe': Sharpe,
            'Sortino': Sortino,
            'AvgDD': AverageDD,
            'MaxDD': MaxDD,
            '% Positive Ret.': PositiveRetRatio
        }

    def compute_metric(self,
                       portoflio_returns: pd.DataFrame,
                       metric_name: str):
        
        torch_portfolio_returns = torch.tensor(portoflio_returns.dropna().values)
        
        metric = self.str_to_metric[metric_name]
        metric = metric()

        portfolio_stat = metric.forward(returns=torch_portfolio_returns).item()

        return portfolio_stat

    def compute_summary_statistics(self,
                                   portfolio_returns: pd.DataFrame,
                                   default_metrics: list = [ExpectedRet, Volatility, Sharpe, Sortino, AverageDD, MaxDD, PositiveRetRatio]):
        
        torch_portfolio_returns = torch.tensor(portfolio_returns.dropna().values)

        portfolio_stats = {}
        for metric in default_metrics:
            metric = metric()
            portfolio_stats[metric.name] = metric.forward(returns=torch_portfolio_returns).item()

        return portfolio_stats