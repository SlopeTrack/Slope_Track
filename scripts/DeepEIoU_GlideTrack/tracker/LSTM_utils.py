import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import generalized_box_iou

import numpy as np
import torch


def normalize_sequence(positions, option, device="cpu"):

    coords = np.array(positions, dtype=np.float32)  # shape: [seq_len, 4]
    ref = coords[0]  # reference frame: first position
    x, y, w, h = coords.T  # unpack each component
    T = coords.shape[0]

    if option == 4:
        # Compute cumulative sum and moving average for height
        cumsum = np.cumsum(h)  # cumulative sum excluding the first element
        time = np.arange(2, h.shape[0] + 1).reshape(-1, 1)  # time steps from 2 to seq_len
        moving_avg_h = (cumsum[1:] / time.squeeze())  # shape: [seq_len - 1]

        cumsum = np.cumsum(w)
        time = np.arange(2, w.shape[0] + 1).reshape(-1, 1)
        moving_avg_w = cumsum[1:] / time.squeeze()

        # Normalize x, y, w, h using moving_avg and reference
        norm = np.stack([
            (x[1:] - ref[0]) / moving_avg_w,
            (y[1:] - ref[1]) / moving_avg_h,
            (w[1:] - ref[2]) / moving_avg_w,
            (h[1:] - ref[3]) / moving_avg_h
        ], axis=1)  # shape: [seq_len - 1, 4]

        # Pad with zeros for the first frame
        norm = np.concatenate([[[0.0, 0.0, 0.0, 0.0]], norm], axis=0)  # shape: [seq_len, 4]

        # Delta: scaling factors used for normalization (includes zero for first frame)
        #delta = np.concatenate([[0.0], moving_avg], axis=0)  # shape: [seq_len]
        delta_w = np.concatenate((w[:1], moving_avg_w), axis=0)
        delta_h = np.concatenate((h[:1], moving_avg_h), axis=0)
        delta = np.stack([delta_w, delta_h], axis=1)


    if option == 2:
        fps=30
        ref_time = 1/int(fps)
        num_sort = np.arange(2, T + 1).reshape(-1, 1)
        r_time = num_sort/fps
        delta = (r_time - ref_time).squeeze()
        #print(delta.shape, x[1:].shape, ref[0].shape) 
        norm = np.stack([(x[1:] - ref[0]) / (delta * w[1:]),
                         (y[1:] - ref[1]) / (delta * h[1:]),
                         (w[1:] - ref[2]) / (delta * w[1:]),
                         (h[1:] - ref[3]) / (delta * h[1:])], axis=1)

        # Pad with zeros for the first frame
        norm = np.concatenate([[[0.0, 0.0, 0.0, 0.0]], norm], axis=0)  # shape: [seq_len, 4]

        # Delta: scaling factors used for normalization (includes zero for first frame)
        delta = np.concatenate([[0.0], delta], axis=0)  # shape: [seq_len]

    # Convert to torch tensors
    norm_tensor = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).to(device)  # shape: [1, seq_len, 4]
    ref_tensor = torch.tensor(ref, dtype=torch.float32).unsqueeze(0).to(device)  # shape: [1, 4]
    delta_tensor = torch.tensor(delta, dtype=torch.float32).unsqueeze(0).to(device)  # shape: [1, seq_len]

    return norm_tensor, ref_tensor, delta_tensor

"""def denormalize_predictions(pred, ref, delta, option):
    
    Denormalize a sequence of predictions.

    Args:
        pred (torch.Tensor): Normalized predictions, shape [batch_size, seq_len, 4].
        ref (torch.Tensor): Reference frame positions, shape [batch_size, 4].
        delta (torch.Tensor): Moving average scaling factors, shape [batch_size, seq_len].

    Returns:
        torch.Tensor: Denormalized predictions, shape [batch_size, seq_len, 4].
    
    #print(pred.shape, ref.shape, delta.shape)
    x_ref, y_ref, w_ref, h_ref = ref[:, 0], ref[:, 1], ref[:, 2], ref[:, 3]

    #print(pred.shape ,x_ref.shape, delta.shape)
    if option == 4:
        mean_h = delta[:, -1, 1]
        #print(delta.shape)
        mean_w = delta[:, -1, 0]
   
        device = pred.device
        dtype = pred.dtype
        seq_len = pred.shape[1]
        denorm_preds = torch.zeros((1, seq_len, 4), device=device, dtype=dtype)

        for c in range(seq_len):

            #print(pred[:, c, 0], mean_w,  x_ref.unsqueeze(1) )

            denorm_preds[:, c, 0] = pred[:, c, 0] * mean_w + x_ref.unsqueeze(1)
            denorm_preds[:, c, 1] = pred[:, c, 1] * mean_h + y_ref.unsqueeze(1)
            denorm_preds[:, c, 2] = pred[:, c, 2] * mean_w + w_ref.unsqueeze(1)
            denorm_preds[:, c, 3] = pred[:, c, 3] * mean_h + h_ref.unsqueeze(1)

            h_new = denorm_preds[:, c, 3]
            w_new = denorm_preds[:, c, 2]

            mean_h = (mean_h * c + h_new) /(c + 1)
            mean_w = (mean_w * c + w_new) /(c + 1)
            print(mean_h,mean_w)

            #w_t = mean_w
            #h_t = mean_h
            #print(denorm_preds[:, c, 0], denorm_preds[:, c, 1])

        x_abs = denorm_preds[:, :, 0]
        y_abs = denorm_preds[:, :, 1]
        w_abs = denorm_preds[:, :, 2]
        h_abs = denorm_preds[:, :, 3]
        #print(denorm_preds)

    if option == 2:
        h_t = h_ref/(1-pred[:, :, 3])
        w_t = w_ref/(1-pred[:, :, 2])
        x_abs = pred[:, :, 0] * (delta[:, -1] * w_t) + x_ref
        y_abs = pred[:, :, 1] * (delta[:, -1] * h_t) + y_ref
        w_abs = pred[:, :, 2] * (delta[:, -1] * w_t) + w_ref
        h_abs = pred[:, :, 3] * (delta[:, -1] * h_t) + h_ref

    output = torch.stack([x_abs, y_abs, w_abs, h_abs], dim=2)
    return output"""







def denormalize_predictions(pred, ref, delta, option):

    """
    Denormalize a sequence of predictions.

    Args:
        pred (torch.Tensor): Normalized predictions, shape [batch_size, seq_len, 4].
        ref (torch.Tensor): Reference frame positions, shape [batch_size, 4].
        delta (torch.Tensor): Moving average scaling factors, shape [batch_size, seq_len].

    Returns:
        torch.Tensor: Denormalized predictions, shape [batch_size, seq_len, 4]."""

    x_ref, y_ref, w_ref, h_ref = ref[:, 0], ref[:, 1], ref[:, 2], ref[:, 3]

    #print(pred.shape ,x_ref.shape, delta.shape)
    if option == 4:
        h_t = delta[:, -1, 1]
        #print(delta.shape)
        w_t = delta[:, -1, 0]

        x_abs = pred[:, :, 0] * w_t + x_ref.unsqueeze(1)
        y_abs = pred[:, :, 1] * h_t + y_ref.unsqueeze(1)
        w_abs = pred[:, :, 2] * w_t + w_ref.unsqueeze(1)
        h_abs = pred[:, :, 3] * h_t + h_ref.unsqueeze(1)
    if option == 2:
        h_t = h_ref/(1-pred[:, :, 3])
        w_t = w_ref/(1-pred[:, :, 2])
        x_abs = pred[:, :, 0] * (delta[:, -1] * w_t) + x_ref
        y_abs = pred[:, :, 1] * (delta[:, -1] * h_t) + y_ref
        w_abs = pred[:, :, 2] * (delta[:, -1] * w_t) + w_ref
        h_abs = pred[:, :, 3] * (delta[:, -1] * h_t) + h_ref

    return torch.stack([x_abs, y_abs, w_abs, h_abs], dim= -1)


def denormalize_sequence(norm, ref, last_abs, args, option, widths, heights, delta=None):
    x_ref, y_ref, w_ref, h_ref = ref[:, 0], ref[:, 1], ref[:, 2], ref[:, 3]


    # Reshape for broadcasting: (B, 1)
    x_ref = x_ref.unsqueeze(1)
    y_ref = y_ref.unsqueeze(1)
    w_ref = w_ref.unsqueeze(1)
    h_ref = h_ref.unsqueeze(1)

    if isinstance(delta, (int, float)):
        delta = torch.full((B, T), float(delta), device=norm.device, dtype=norm.dtype)
    elif delta.dim() == 1:
        delta = delta.unsqueeze(1).expand(-1, T)  # (B, 1) -> (B, T)
    h_t = delta[:, :-args.target_len]
    # print('Delta denorm', h_t, x_ref)
    # print(norm[:, :, 0])
    x_abs = (norm[:, :, 0].to(device) * h_t) + x_ref
    y_abs = (norm[:, :, 1].to(device) * h_t) + y_ref
    w_abs = (norm[:, :, 2].to(device) * h_t) + w_ref
    h_abs = (norm[:, :, 3].to(device) * h_t) + h_ref
    return torch.stack([x_abs, y_abs, w_abs, h_abs], dim=-1)  # (B, T, 4)

def denormalize(norm, ref, last_abs, args, option, widths, heights, delta=None):
    x_ref, y_ref, w_ref, h_ref = ref[:, 0], ref[:, 1], ref[:, 2], ref[:, 3]
    x_abs = norm[:, :, 0].to(device) * (delta[:, -args.target_len:]) + x_ref.unsqueeze(1)
    y_abs = norm[:, :, 1].to(device) * (delta[:, -args.target_len:]) + y_ref.unsqueeze(1)
    w_abs = norm[:, :, 2].to(device) * (delta[:, -args.target_len:]) + w_ref.unsqueeze(1)
    h_abs = norm[:, :, 3].to(device) * (delta[:, -args.target_len:]) + h_ref.unsqueeze(1)
    return torch.stack([x_abs, y_abs, w_abs, h_abs], dim=2)

