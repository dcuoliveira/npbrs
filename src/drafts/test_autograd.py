import torch
from strategies.RobustEWMA import RobustEWMA
from estimators.DependentBootstrapSampling import DependentBootstrapSampling
from strategies.SimpleNN import SimpleNN
from strategies.NNRobustStrategicControl import NNRobustStrategicControl


wsize = 60
time_series = torch.randn(100,10)
N,M = time_series.shape
print(str(N) + " " + str(M))
boot_method = DependentBootstrapSampling(time_series = time_series,boot_method = "cbb")
nn_model = SimpleNN(wsize=wsize)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(nn_model.parameters(),lr = 0.0002,betas = (0.5,0.999))
#
Strategy = NNRobustStrategicControl(nn_model,loss_fn,optimizer,boot_method)
Strategy.optimize()