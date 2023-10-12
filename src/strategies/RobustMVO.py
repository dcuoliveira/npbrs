import torch
import numpy as np
from estimators.Estimators import Estimators
from strategies.RobustStrategicControl import RobustStrategicControl
from estimators.DependentBootstrapSampling import DependentBootstrapSampling

class RobustMVO(RobustStrategicControl):
    #
    def __init__(self,
                 time_series: torch.Tensor,
                 boot_method: DependentBootstrapSampling,
                 risk_aversion: float,
                 step: float = 1e-2,
                 tol: float = 1e-6,
                 B: int = 1000,
                 alpha: float = 0.25,
                 num_iter: int = 1000
                 ) -> None:
        self.risk_aversion = risk_aversion
        self.step = step
        self.tol = tol
        super(RobustStrategicControl, self).__init__(time_series, boot_method, B,alpha,num_iter)
    # 

    # Implement methods with Polymorphism
    
    # Module to initialize parameters
    def init_parameters(self) -> dict:
        K = self.time_series.shape[1]
        theta = theta = torch.Tensor(np.random.uniform(-1, 1, size = K))
        parameters = {"theta":theta}
        return parameters
    
    # Module to compute utility given
    def compute_utility(self,boot_sample:torch.Tensor,
                        parameters:dict) -> dict:
        mean = Estimators.MLEMean(boot_sample)
        cov = Estimators.MLECovariance(boot_sample)
        #
        theta = parameters["theta"]
        utility = torch.matmul(theta,mean) - self.risk_aversion*torch.matmul(theta,torch.matmul(cov,theta))

        # utility dict
        utility_dict = {"value": utility,"theta":theta,"mean":mean,"cov":cov}
        #
        return utility_dict

    # Module to compute utilities
    # returns a dict
    def evaluate_bootstrap_utility(self,
                                   bootstrap_utilities:list) -> dict:
        sorted_bootstrap_utilities = sorted(bootstrap_utilities, key=lambda utility_dict: utility_dict["value"])
        # select
        selected_bootstrap_utilities = sorted_bootstrap_utilities[int(self.B*self.alpha)]
        parameters = {"theta":selected_bootstrap_utilities["theta"],
                      "mean":selected_bootstrap_utilities["mean"],
                      "cov":selected_bootstrap_utilities["cov"]}
        return parameters   
    
    # Module to update the parameters
    # returns the dict with the update parameters
    def update_parameters(self,
                          parameters: dict) -> dict:
        theta = parameters["theta"]
        dtheta = parameters["mean"] - 2*self.risk_aversion*torch.matmul(parameters["cov"],theta)
        new_theta = theta + self.step*dtheta
        new_parameters = {"theta":new_theta, "mean": parameters["mean"],"cov":parameters["cov"]}

        return new_parameters

    # Module to check if the parameter search already converged
    def converged(self,
                  parameters: dict,
                  new_parameters: dict):
        diff = torch.sum(torch.abs(parameters["theta"] - new_parameters["theta"]))
        if diff < self.tol:
            return True
        else:
            return False
