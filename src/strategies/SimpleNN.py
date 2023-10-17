import torch
import torch.nn as nn
import torch

class SimpleNN(nn.Module):
    #
    def __init__(self,
                 wsize: int = 60
                 ) -> None:
        super(SimpleNN,self).__init__()
        self.wsize = wsize
        self.layer = nn.Linear(wsize,1)
    
    # forward
    def forward(self,X):
        out = self.layer(X) #torch.nn.functional.sigmoid(self.layer(X))
        return out

    # return windows size
    def get_wsize(self) -> int:
        return self.wsize