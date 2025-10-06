import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mamba_ssm import Mamba2, Mamba
from torchvision.ops import generalized_box_iou
import torch
import torch.nn as nn
from torchvision.ops import complete_box_iou_loss, generalized_box_iou_loss


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert (cx, cy, w, h) format to (x1, y1, x2, y2).
    
    Args:
        boxes (Tensor): shape (..., 4) in (cx, cy, w, h) format
    Returns:
        Tensor: shape (..., 4) in (x1, y1, x2, y2) format
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)



def ciou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    # CIoU loss is already implemented in torchvision
    
    pred_boxes = xywh_to_xyxy(pred_boxes)
    target_boxes = xywh_to_xyxy(target_boxes)

    loss = complete_box_iou_loss(pred_boxes, target_boxes, reduction='none')
    return loss

def giou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    # CIoU loss is already implemented in torchvision
    
    pred_boxes = xywh_to_xyxy(pred_boxes)
    target_boxes = xywh_to_xyxy(target_boxes)

    loss = generalized_box_iou_loss(pred_boxes, target_boxes, reduction='mean')
    return loss


def bbox_iou(box1, box2, eps=1e-6):
    """
    box1, box2: tensors of shape [N, 4] in [cx, cy, w, h] format
    Returns: IoU for each pair
    """
    # Convert to [x1, y1, x2, y2]
    box1_xy = torch.cat([box1[:,:, :2] - box1[:,:, 2:] / 2, box1[:,:,:2] + box1[:,:, 2:] / 2], dim=-1)
    box2_xy = torch.cat([box2[:,:, :2] - box2[:,:, 2:] / 2, box2[:,:, :2] + box2[:, :, 2:] / 2], dim=-1)

    x1 = torch.max(box1_xy[:,:, 0], box2_xy[:,:,0])
    y1 = torch.max(box1_xy[:,:, 1], box2_xy[:,:, 1])
    x2 = torch.min(box1_xy[:,:, 2], box2_xy[:,:, 2])
    y2 = torch.min(box1_xy[:,:, 3], box2_xy[:,:, 3])

    inter_area = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    area1 = (box1_xy[:,:, 2] - box1_xy[:,:, 0]) * (box1_xy[:,:, 3] - box1_xy[:,:, 1])
    area2 = (box2_xy[:,:, 2] - box2_xy[:,:, 0]) * (box2_xy[:,:, 3] - box2_xy[:,:, 1])

    union_area = area1 + area2 - inter_area + eps
    iou = inter_area / union_area
    return iou


def euclid(pred, target):

    diff = pred-target
    distances = torch.norm(diff, dim=1)
    return distances
    
    
def nll_loss1(target, mu, sigma, rho, eps=1e-6):
    """
    target: (B, 2) - true x, y
    mu: (B, 2) - predicted mean x, y
    sigma: (B, 2) - predicted std x, y (must be > 0)
    rho: (B,) - predicted correlation (must be in [-1, 1])
    """
    x, y = target[:, 0], target[:, 1]
    mu_x, mu_y = mu[:, 0], mu[:, 1]
    sigma_x, sigma_y = sigma[:, 0], sigma[:, 1]

    # z term in bivariate Gaussian
    z_x = (x - mu_x) / (sigma_x + eps)
    z_y = (y - mu_y) / (sigma_y + eps)

    z = z_x ** 2 + z_y ** 2 - 2 * rho * z_x * z_y
    denom = 2 * (1 - rho ** 2 + eps)

    # Normalization constant
    norm_const = 2 * torch.pi * sigma_x * sigma_y * torch.sqrt(1 - rho ** 2 + eps)

    #print(x,y, mu_y, mu_x,sigma_x, sigma_y,  z_x, z_y, z, denom, norm_const)
    log_prob = -z / denom - torch.log(norm_const + eps)

    return -log_prob.mean()  # NLL loss

def nll_loss2(pred_mean, target, var):
    criterion = nn.GaussianNLLLoss()
    return criterion(pred_mean, target, var)

def mse_loss(pred_mean, target):
    criterion = nn.MSELoss()
    return criterion(pred_mean, target)

def l1_loss(pred_mean, target):
    criterion = nn.SmoothL1Loss(beta=0.001, reduction='mean')
    return criterion(pred_mean, target)
