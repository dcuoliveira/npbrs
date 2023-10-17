import torch
import numpy as np
from utils.dataset_utils import timeseries_sliding_window_train
from estimators.Estimators import Estimators
from estimators.DependentBootstrapSampling import DependentBootstrapSampling
# Class for Robust Strategic Control
# This class contains the different ways to generate time series samplings using bootstrap

class NNRobustStrategicControl:

    # init everything
    def __init__(self,
                 nn_model: torch.nn,
                 loss_fn: torch.nn.functional,
                 optimizer: torch.optim,
                 boot_method: DependentBootstrapSampling,
                 B: int = 1000,
                 alpha: float = 0.25,
                 n_epochs: int = 2000
                 ) -> None:
        # nn_model: neural network model to train (input is of size wsize)
        # loss_fn: loss function
        # boot_method:  bootrap method to generate the random sampling
        # B: UNumber of bootstraps to obtain
        # n_epochs: number of epochs for in the training step
        super().__init__()
        self.wsize = nn_model.get_wsize()
        self.optimizer = optimizer
        self.nnetwork = nn_model
        self.loss_fn = loss_fn
        self.boot_method = boot_method
        self.B = B
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.boot_time_series = self.create_boot_time_series()

    # procedure to create bootstraped time series
    def create_boot_time_series(self) -> list:
        list_boot_time_series = list()
        for _ in range(self.B):
            boot_sample = self.boot_method.sample()
            list_boot_time_series.append(timeseries_sliding_window_train(boot_sample,self.wsize))
        #
        return list_boot_time_series
    

    # optimize
    def optimize(self) -> None:
        # number of epochs
        for _ in range(self.n_epochs):
            self.optimizer.zero_grad()
            boot_utilities = list()
            for idx_boot in range(len(self.boot_time_series)):
                X_train = self.boot_time_series[idx_boot]["X"]
                Y_train = self.boot_time_series[idx_boot]["Y"]
                #
                output = self.nnetwork(X_train)
                # computing the utility for the bootstraped time series
                utility = self.loss_fn(output,Y_train)
                # save
                boot_utilities.append({"value": utility,"idx": idx_boot})
            #
            sorted_bootstrap_utilities = sorted(boot_utilities, key=lambda utility_dict: utility_dict["value"])
            # select
            selected_bootstrap_utilities = sorted_bootstrap_utilities[int(len(boot_utilities)*self.alpha)]
            # get the required bootstrap \alpha-quantile
            best_idx = selected_bootstrap_utilities["idx"] 
            #
            X_train = self.boot_time_series[best_idx]["X"]
            Y_train = self.boot_time_series[best_idx]["Y"]
            # compute the loss function
            output = self.nnetwork(X_train)
            # backpropagate the loss function
            loss = self.loss_fn(output,Y_train)
            print(loss)
            loss.backward()
            self.optimizer.step()
