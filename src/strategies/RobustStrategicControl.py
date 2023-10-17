
import torch
import numpy as np
from estimators.Estimators import Estimators
from estimators.DependentBootstrapSampling import DependentBootstrapSampling
# Class for Robust Strategic Control
# This class contains the different ways to generate time series samplings using bootstrap

class RobustStrategicControl:

    def __init__(self,
                 time_series: torch.Tensor,
                 boot_method: DependentBootstrapSampling,
                 B: int = 1000,
                 alpha: float = 0.25,
                 num_iter: int = 1000
                 ) -> None:
        # time_series: assets time series
        # boot_method:  bootrap method to generate the random sampling
        # Utility: Utility function to evaluate the strategies
        # Utility_derivative: Utility derivative to update the parameters of interest
        # alpha: The Confidence interval to optimize
        super().__init__()
        self.time_series = time_series
        self.boot_method = boot_method
        self.B = B
        self.alpha = alpha
        self.num_iter = num_iter
        self.parameters = None # parameters to optimize 


    # algorithm
    def forward(self) -> dict:
        # create bootstraped time series
        bootstrap_samples = list()
        for _ in range(self.B):
            bootstrap_samples.append(self.boot_method.sample())
        # initialize theta
        parameters = self.init_parameters()
        idx_iter = 0
        while idx_iter < self.num_iter:
            # compute utilities for all bootstraps
            bootstrap_utilities = list()
            for idx in range(len(bootstrap_samples)):
                sample_utility = self.compute_utility(bootstrap_samples[idx],parameters)
                bootstrap_utilities.append(sample_utility)
            # sort utilities with index
            selected_parameters = self.evaluate_bootstrap_utility(bootstrap_utilities)
            new_parameters = self.update_parameters(selected_parameters)
            if self.converged(parameters,new_parameters):
                print("convergence achieved")
                break
            #
            parameters = new_parameters
            self.num_iter = self.num_iter - 1
        #
        return parameters                   
    
    # Module to initialize parameters
    def init_parameters(self) -> dict:
        None
    
    # Module to compute utility given
    def compute_utility(self,boot_sample:torch.Tensor,
                        parameters:dict) -> dict:
        None

    # Module to compute utilities
    # returns a dict
    def evaluate_bootstrap_utility(self,
                                   bootstrap_utilities:list) -> dict:
        None
    
    # Module to update the parameters
    # returns the dict with the update parameters
    def update_parameters(self,
                          parameters: dict) -> dict:
        None

    # Module to check if the parameter search already converged
    def converged(self,
                  parameters: dict,
                  new_parameters: dict) -> bool:
        None