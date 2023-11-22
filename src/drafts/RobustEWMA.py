import torch
import numpy as np
from estimators.Estimators import Estimators
from strategies.RobustStrategicControl import RobustStrategicControl
from estimators.DependentBootstrapSampling import DependentBootstrapSampling

class RobustEWMA(RobustStrategicControl):
    #
    def __init__(self,
                 time_series: torch.Tensor,
                 boot_method: DependentBootstrapSampling,
                 windows_size: float = 50,
                 step: float = 1e-2,
                 tol: float = 1e-6,
                 B: int = 1000,
                 alpha: float = 0.25,
                 num_iter: int = 1000
                 ) -> None:
        self.windows_size = windows_size
        self.step = step
        self.tol = tol
        super(RobustEWMA, self).__init__(time_series, boot_method, B,alpha,num_iter)
    # 

    # Implement methods with Polymorphism
    
    # Module to initialize parameters
    def init_parameters(self) -> dict:
        self.alpha = torch.randn(1)
        parameters = {"alpha":self.alpha}
        return parameters
    
    # Module to compute utility given
    def compute_utility(self,boot_sample:torch.Tensor,
                        parameters:dict) -> dict:
        N,M = boot_sample.shape
        alpha = parameters["alpha"]
        # build the EWMA
        sliding_window = torch.ones(self.windows_size)
        for i in range(self.windows_size-2,-1,-1):
            sliding_window[i] = sliding_window[i + 1]*alpha
        #
        sliding_window = (1 - alpha.item())*sliding_window
        EWMA_time_series = torch.Tensor(N - self.windows_size + 1,M)
        # compute cost
        for j in range(M):
            single_time_series = self.time_series[:,j]
            EWMA = torch.conv1d(input = single_time_series.view(1,1,-1),weight = sliding_window.flip(0).view(1,1,-1),padding = 0).squeeze()
            EWMA_time_series[:,j] = EWMA
        # utility is the return function
        utility = torch.mean(EWMA_time_series)

        # utility dict
        utility_dict = {"alpha": alpha,"value": utility}
        #
        return utility_dict

    # Module to compute utilities
    # returns a dict
    def evaluate_bootstrap_utility(self,
                                   bootstrap_utilities:list) -> dict:
        sorted_bootstrap_utilities = sorted(bootstrap_utilities, key=lambda utility_dict: utility_dict["value"])

        # select
        selected_bootstrap_utilities = sorted_bootstrap_utilities[int(len(bootstrap_utilities)*self.alpha)]
        print(selected_bootstrap_utilities)
        parameters = {"alpha":selected_bootstrap_utilities["alpha"],"value":selected_bootstrap_utilities["value"]}
        return parameters   
    
    # Module to update the parameters
    # returns the dict with the update parameters
    def update_parameters(self,
                          parameters: dict) -> dict:
        N,M = self.time_series.shape
        alpha = parameters["alpha"]
        # build the EWMA
        sliding_window = torch.ones(self.windows_size)
        sliding_window2 = torch.ones(self.windows_size)
        for i in range(self.windows_size-2,-1,-1):
            sliding_window[i] = sliding_window[i + 1]*alpha

        for i in range(self.windows_size-3,-1,-1):
            sliding_window2[i] = sliding_window2[i + 1]*alpha
        
        for i in range(self.windows_size-1,-1,-1):
            sliding_window2[i] = i*sliding_window2[i]
        #
        # computing derivative
        sliding_window2 = (1 - alpha)*sliding_window2
        EWMA_time_series = torch.Tensor(N - self.windows_size + 1,M)
        # compute cost
        for j in range(M):
            single_time_series = self.time_series[:,j]
            EWMA = torch.conv1d(input = single_time_series.view(1,1,-1),weight = sliding_window.flip(0).view(1,1,-1),padding = 0).squeeze()
            dEWMA = torch.conv1d(input = single_time_series.view(1,1,-1),weight = sliding_window2.flip(0).view(1,1,-1),padding = 0).squeeze()

            EWMA_time_series[:,j] = dEWMA - EWMA 
        # utility is the return function
        utility = torch.mean(EWMA_time_series)

        new_parameters = {"utility":utility,"alpha":(alpha - self.step*utility)}

        return new_parameters

    # Module to check if the parameter search already converged
    def converged(self,
                  parameters: dict,
                  new_parameters: dict):
        diff = torch.mean(torch.abs(parameters["alpha"] - new_parameters["alpha"]))
        if diff < self.tol:
            return True
        else:
            return False
