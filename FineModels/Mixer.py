from torch import nn
import torch
import torch.nn.functional as F
from functools import partial
from einops.layers.torch import Rearrange, Reduce

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(*, timeframe, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
    """_summary_

    Args:
        timeframe (_type_): _description_
        channels (_type_): _description_
        patch_size (_type_): _description_
        dim (_type_): Dimension to reduce each patch to
        depth (_type_): Number of Mixer Layers (N)
        num_classes (_type_): _description_
        expansion_factor (int, optional): Inside the FeedForward, how much we are enlarging the input. Defaults to 4.
        expansion_factor_token (float, optional): _description_. Defaults to 0.5.
        dropout (_type_, optional): _description_. Defaults to 0..

    Returns:
        _type_: Sequential model
    """
    assert timeframe % patch_size == 0, 'timeframe must be divisible by patch size'
    num_patches = timeframe // patch_size
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (p l) -> b l (p c)', p = patch_size),
        nn.Linear(patch_size * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        nn.BatchNorm1d(num_patches),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, dim//2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(dim//2, num_classes)
    )