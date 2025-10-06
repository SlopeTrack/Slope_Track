import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mamba_ssm import Mamba2, Mamba
from torchvision.ops import generalized_box_iou

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def denormalize_sequence(norm, ref, last_abs, args, option, widths, heights, delta=None):
    """
    Denormalize normalized bounding box sequences.
    
    Args:
        norm (Tensor): (B, T, 4) - normalized coordinates
        ref (Tensor): (B, 4) - reference frame
        last_abs (Tensor): (B, 4) - last absolute frame
        option (int): normalization mode (1 or 2/3)
        delta (float or Tensor): scalar, (B,) or (B, T), required for option 2/3ù
    
    Returns:
        Tensor: (B, T, 4) - denormalized coordinates
    """
    #B, T, _ = norm.shape
    if option != 5:
        x_ref, y_ref, w_ref, h_ref = ref[:, 0], ref[:, 1], ref[:, 2], ref[:, 3]
        #w_t, h_t = last_abs[:, 2], last_abs[:, 3]
        #print(w_t, h_t)

        # Reshape for broadcasting: (B, 1)
        x_ref = x_ref.unsqueeze(1)
        y_ref = y_ref.unsqueeze(1)
        w_ref = w_ref.unsqueeze(1)
        h_ref = h_ref.unsqueeze(1)
        #w_t = w_t.unsqueeze(1)
        #h_t = h_t.unsqueeze(1)

    widths = torch.from_numpy(widths).float().to(norm.device)
    heights = torch.from_numpy(heights).float().to(norm.device)
    #print(norm[:, :, 0].shape, ref[:, 0].shape)
    if option == 1:
        h_t = h_ref/(1-norm[:, :, 3])
        w_t = w_ref/(1-norm[:, :, 2])
        #print(norm[:, :, 0] * w_t, x_ref)
        x_abs = (norm[:, :, 0] * w_t) + x_ref
        y_abs = (norm[:, :, 1]  * h_t) + y_ref
        w_abs = (norm[:, :, 2] * w_t) + w_ref
        h_abs = (norm[:, :, 3] * h_t) + h_ref
    elif option == 2:
        if delta is None:
            raise ValueError("delta is required for option 2 or 3")
        # Normalize delta to shape (B, T)
        if isinstance(delta, (int, float)):
            delta = torch.full((B, T), float(delta), device=norm.device, dtype=norm.dtype)
        elif delta.dim() == 1:
            delta = delta.unsqueeze(1).expand(-1, T)  # (B, 1) -> (B, T)

        #print(delta.shape)
        h_t = h_ref/(1-norm[:, :, 3])
        w_t = w_ref/(1-norm[:, :, 2])
        #print(delta.shape)
        x_abs = norm[:, :, 0] * (delta * w_t) + x_ref
        y_abs = norm[:, :, 1] * (delta * h_t) + y_ref
        w_abs = norm[:, :, 2] * (delta * w_t) + w_ref
        h_abs = norm[:, :, 3] * (delta * h_t) + h_ref
    elif option == 3:
        h_t = h_ref/(1-norm[:, :, 3])
        #print(norm[:, :, 0] * w_t, x_ref)
        x_abs = (norm[:, :, 0] * h_t) + x_ref
        y_abs = (norm[:, :, 1]  * h_t) + y_ref
        w_abs = (norm[:, :, 2] * h_t) + w_ref
        h_abs = (norm[:, :, 3] * h_t) + h_ref
    elif option == 4:
        #h_t = h_ref/(1-norm[:, :, 3])
        """if isinstance(delta, (int, float)):
            delta = torch.full((B, T), float(delta), device=norm.device, dtype=norm.dtype)
        elif delta.dim() == 1:
            delta = delta.unsqueeze(1).expand(-1, T)"""  # (B, 1) -> (B, T)
        h_t = delta[:, :, 1]
        w_t = delta[:, :, 0]
        #print('Delta denorm', h_t[7], x_ref[7])
        #print(norm[:, :, 0])
        x_abs = (norm[:, :, 0].to(device) * w_t) + x_ref
        y_abs = (norm[:, :, 1].to(device)  * h_t) + y_ref
        w_abs = (norm[:, :, 2].to(device) * w_t) + w_ref
        h_abs = (norm[:, :, 3].to(device) * h_t) + h_ref
    elif option == 5:
        x_abs = norm[:, :, 0] * widths
        y_abs = norm[:, :, 1] * heights
    elif option == 6:
        x_abs = norm[:, :, 0] * widths
        y_abs = norm[:, :, 1] * heights
        w_abs = norm[:, :, 2] * widths
        h_abs = norm[:, :, 3] * heights
    elif option== 7:
        if delta is None:
            raise ValueError("delta is required for option 2 or 3")
        # Normalize delta to shape (B, T)
        if isinstance(delta, (int, float)):
            delta = torch.full((B, T), float(delta), device=norm.device, dtype=norm.dtype)
        elif delta.dim() == 1:
            delta = delta.unsqueeze(1).expand(-1, T)  # (B, 1) -> (B, T)

        #print(delta.shape)
        #h_t = h_ref/(1-norm[:, :, 3])
        #w_t = w_ref/(1-norm[:, :, 2])
        x_abs = norm[:, :, 0] * delta + x_ref
        y_abs = norm[:, :, 1] * delta + y_ref
        w_abs = norm[:, :, 2] * delta + w_ref
        h_abs = norm[:, :, 3] * delta + h_ref
    if option == 5:
        return torch.stack([x_abs, y_abs], dim=-1)
    else:
        return torch.stack([x_abs, y_abs, w_abs, h_abs], dim=-1)  # (B, T, 4)


def denormalize(norm, ref, last_abs, args, option, widths, heights, delta=None):
    if option != 5:
        x_ref, y_ref, w_ref, h_ref = ref[:, 0], ref[:, 1], ref[:, 2], ref[:, 3]
        #w_t, h_t = last_abs[:,2], last_abs[:,3] #norm[:, 2], norm[:, 3]
    widths = torch.from_numpy(widths).float().to(norm.device)
    heights = torch.from_numpy(heights).float().to(norm.device)

    if option == 1:
        h_t = h_ref/(1-norm[:, 3])
        w_t = w_ref/(1-norm[:, 2])
        x_abs = (norm[:, 0] * w_t) + x_ref
        y_abs = (norm[:, 1] * h_t) + y_ref
        w_abs = (norm[:, 2] * w_t) + w_ref
        h_abs = (norm[:, 3] * h_t) + h_ref
    elif option == 2:
        #valid_counts = (delta!= -2).sum(dim=1)
        #delta =(valid_counts + 1).float()/30 - 1/30
        #print(norm[:, 0].shape, delta[:, -1])
        h_t = h_ref.unsqueeze(1)/(1-norm[:,:, 3])
        w_t = w_ref.unsqueeze(1)/(1-norm[:,:, 2])
        x_abs = norm[:, :, 0] * (delta[:, -args.target_len:] * w_t) + x_ref.unsqueeze(1)
        y_abs = norm[:, :, 1] * (delta[:, -args.target_len:] * h_t) + y_ref.unsqueeze(1)
        w_abs = norm[:, :, 2] * (delta[:, -args.target_len:] * w_t) + w_ref.unsqueeze(1)
        h_abs = norm[:, :, 3] * (delta[:, -args.target_len:] * h_t) + h_ref.unsqueeze(1)
    elif option == 3:
        h_t = h_ref.unsqueeze(1)/(1-norm[:,:, 3])
        #w_t = w_ref/(1-norm[:, 2])
        x_abs = (norm[:,:, 0] * h_t) + x_ref.unsqueeze(1)
        y_abs = (norm[:,:, 1] * h_t) + y_ref.unsqueeze(1)
        w_abs = (norm[:,:, 2] * h_t) + w_ref.unsqueeze(1)
        h_abs = (norm[:,:, 3] * h_t) + h_ref.unsqueeze(1)
    elif option == 4:
        #print('Norm and delta', norm[:, :, 0].shape, delta[:, -2:].shape, x_ref.shape)
        #print('Delta denorm', delta.shape, x_ref.shape, norm.shape)
        h_t = delta[:, :, 1]
        #print(delta.shape)
        w_t = delta[:, :, 0]
        x_abs = norm[:, :, 0].to(device) * w_t + x_ref.unsqueeze(1)
        y_abs = norm[:, :, 1].to(device) * h_t + y_ref.unsqueeze(1)
        w_abs = norm[:, :, 2].to(device) * w_t + w_ref.unsqueeze(1)
        h_abs = norm[:, :, 3].to(device) * h_t + h_ref.unsqueeze(1)
        #print(x_abs.shape)
    elif option == 5:
        x_abs = norm[:,:, 0] * widths
        y_abs = norm[:,:, 1] * heights
    elif option == 6:
        x_abs = norm[:, 0] * widths.squeeze()
        y_abs = norm[:, 1] * heights.squeeze()
        w_abs = norm[:, 2] * widths.squeeze()
        h_abs = norm[:, 3] * heights.squeeze()
    elif option == 7:
        x_abs = norm[:, 0] * (delta[:, -1]) + x_ref
        y_abs = norm[:, 1] * (delta[:, -1]) + y_ref
        w_abs = norm[:, 2] * (delta[:, -1]) + w_ref
        h_abs = norm[:, 3] * (delta[:, -1]) + h_ref
    #print('Size of denorm', torch.stack([x_abs, y_abs, w_abs, h_abs], dim=-1).shape)
    if option == 5:
        return torch.stack([x_abs, y_abs], dim=2)
    else:
        return torch.stack([x_abs, y_abs, w_abs, h_abs], dim=2)


