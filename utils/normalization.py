import torch
import pandas as pd

def normalize(x):
    if isinstance(x, pd.DataFrame):
        x = torch.tensor(x.values, dtype=torch.float32)

    min_vals = x.min(0, keepdim=True).values
    max_vals = x.max(0, keepdim=True).values

    scale = max_vals - min_vals
    scale[scale == 0] = 1

    return 2 * (x - min_vals) / scale - 1


def denormalize(x_norm, x_original):
    if isinstance(x_original, pd.DataFrame):
        x_original = torch.tensor(x_original.values, dtype=torch.float32)

    min_vals = x_original.min(0, keepdim=True).values
    max_vals = x_original.max(0, keepdim=True).values

    return ((x_norm + 1) / 2) * (max_vals - min_vals) + min_vals