"""
N = Batch Size
C = Number of Channels
L = Length of Time series
Hyperparameters to train: Encoder depth, dilation rates, decoder channel, timeframe and offset
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import pandas as pd

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        super(SegmentationHead, self).__init__()
        conv2d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        #upsampling = nn.Upsample(scale_factor=upsampling, mode='linear') if upsampling > 1 else nn.Identity()
        #activation = nn.Softmax(dim=1)
        self.segment = nn.Sequential(conv2d)
        self.linear = nn.Linear(90, 1)
    def forward(self, x):
        x = self.segment(x)
        x = self.linear(x.squeeze())
        return x

class DeepLabV3(nn.Module):
    def __init__(
        self,
        encoder_depth:int = 1, #From [0,4]
        decoder_channels: int = 256,
        in_channels: int = 60,
        classes: int = 1,
        atrous_rates : tuple = (12, 24, 36)
    ): 
        super().__init__()

        self.encoder = Encoder(
            in_channels = in_channels,
            depth = encoder_depth,
        )
        
        self.decoder = DeepLabV3Decoder(
            in_channels=self.encoder.out_channels[-1],
            out_channels=decoder_channels,
            atrous_rates=atrous_rates
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            kernel_size=1,
            upsampling= (encoder_depth**2 if encoder_depth!= 1 else 2),
        )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        features = self.encoder(x) 

        decoder_output = self.decoder(features)

        masks = self.segmentation_head(decoder_output)

        return masks



class DeepLabV3Decoder(nn.Sequential):
    def __init__(self, in_channels, out_channels=256, atrous_rates=(12, 24, 36)):
        super().__init__(
            ASPP(in_channels, out_channels, atrous_rates),
            nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        self.out_channels = out_channels

    def forward(self, *features):
        return super().forward(features[-1])
        

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, separable=False):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            )
        )

        rate1, rate2, rate3 = tuple(atrous_rates)

        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv1d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        
        return self.project(res)

class Encoder(nn.Module):
    def __init__(self,in_channels, depth):
        super(Encoder,self).__init__()
        modules = []
        self.out_channels = [int(i*in_channels*(2**0.5)) for i in range(1,depth+1)]
        self.out_channels.insert(0, in_channels)
        for i in range(depth):
           modules.append(
                nn.Sequential(
                    nn.Conv1d(self.out_channels[i], self.out_channels[i+1], 1, bias=False, stride = 2),
                    nn.BatchNorm1d(self.out_channels[i+1]),
                    nn.ELU(),
                )
           )
        self.mods = nn.ModuleList(modules)
            
    def forward(self,x):
        for mod in self.mods:
            x = mod(x)
        return x
            
    
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-1:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="linear", align_corners=False)
        
from torch.utils.data import Dataset

class TimeFrameDataset(Dataset):
    
    def __init__(self, train_dir:str, timeframe:int, target_dir:str):
        self.timeframe = timeframe
        df = pd.read_parquet(train_dir)
        self.len = len(df)//timeframe
        with open(target_dir, "r") as file:
            read_list = [int(line.strip()) for line in file]
        assert len(df)%self.timeframe == 0, 'The data is not formatted correctly'
        assert self.len == len(read_list), 'The data is not formatted correctly'
        self.labels = torch.FloatTensor(read_list)
        self.features = torch.FloatTensor(df.drop(columns ='state').values).transpose(0,1)
        del df

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        idx_start = self.timeframe * idx
        idx_end = self.timeframe * (idx+1)
        label =  self.labels[idx]
        features = self.features[:,idx_start:idx_end]
        return features, label
    
    
