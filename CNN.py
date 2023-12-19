import torch.nn as nn
import torch
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,in_channels, depth, dropout):
        super(Encoder,self).__init__()
        modules = []
        self.out_channels = [int(i*in_channels*(3**0.5)) for i in range(1,depth+1)]
        self.out_channels.insert(0, in_channels)
        for i in range(depth):
           modules.append(
                nn.Sequential(
                    nn.Conv1d(self.out_channels[i], self.out_channels[i+1], 9, bias=True, stride = 2, padding = 4),
                    nn.BatchNorm1d(self.out_channels[i+1]),
                    nn.ELU(),
                )
           )
        self.mods = nn.ModuleList(modules)
        self.linear1 = nn.Linear(45, 1)
        self.linears = nn.Sequential(nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(225,150),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(150,75),      
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(75,1)) 
  
            
    def forward(self,x):
        for mod in self.mods:
            x = mod(x)
        x = self.linear1(x)
        x = self.linears(x.squeeze())
        return x.squeeze()