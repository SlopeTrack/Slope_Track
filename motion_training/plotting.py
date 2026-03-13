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

def plot_pred(tgt, mean_abs, inp_abs, start_idx, end_idx, target_idx, args, frame_num, sequence, old_track_id, sigma=None, rho=None):

    sample = 7  # index of the sample to visualize

    # Extract indices
    start_idx = start_idx[sample].cpu().numpy()
    end_idx = end_idx[sample].cpu().numpy()
    target_idx = target_idx[sample].cpu().numpy()
    input_len = end_idx - start_idx
    pred_len = tgt.shape[1]  # Number of predicted steps

    # Extract arrays
    input_xy = inp_abs[sample, :input_len, :2].cpu().numpy()       # (input_len, 2)
    pred_xy = mean_abs[sample, :pred_len, :2].cpu().numpy()        # (T, 2)
    tgt_xy  = tgt[sample, :pred_len, :2].cpu().numpy()             # (T, 2)
    frame_numbers = frame_num[sample].cpu().numpy()

    # Plot
    plt.figure(figsize=(7, 7))
    plt.plot(input_xy[:, 0], input_xy[:, 1], 'ko-', label='Input Trajectory')
    plt.plot(pred_xy[:, 0], pred_xy[:, 1], 'bx--', label='Predicted Trajectory')
    plt.plot(tgt_xy[:, 0], tgt_xy[:, 1], 'ro-', label='Ground Truth Targets')

    print('Input',inp_abs[sample, :input_len, :2].cpu().numpy(), 'Mean', mean_abs[sample,:pred_len, :4].cpu().numpy(), 'Target', tgt[sample, :pred_len, :4].cpu().numpy())
    print(sequence[sample], old_track_id[sample], start_idx, end_idx, target_idx) 

    # Annotate frame numbers: input + predicted + ground truth
    for i, (x, y) in enumerate(input_xy):
        if i < len(frame_numbers):
            plt.annotate(str(frame_numbers[i]), (x, y), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=7)

    for t, (px, py) in enumerate(pred_xy):
        plt.annotate(f'P{t}', (px, py), textcoords="offset points", xytext=(5, -10), ha='center', fontsize=7, color='blue')
    
    for t, (tx, ty) in enumerate(tgt_xy):
        plt.annotate(f'T{t}', (tx, ty), textcoords="offset points", xytext=(-10, 5), ha='right', fontsize=7, color='red')

    # Title and info
    plt.title(f"Trajectory Prediction (Multi-step)\nSeq {sequence[sample]}, ID {old_track_id[sample]}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"trajectory_prediction_option_{args.model}_{args.option}_{args.min_len}_{args.max_len}_{args.target_len}_paper_model.png")
    plt.close()
    print("Example prediction:", f"trajectory_prediction_option_{args.model}_{args.option}_{args.min_len}_{args.max_len}_{args.target_len}_paper_model.png")

def plot_train_val(train_losses, val_losses, lrs, args):
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, lrs, label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.title('Learning Rate')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"training_curves_option_{args.model}_{args.option}_{args.min_len}_{args.max_len}_{args.target_len}_paper_model.png", dpi=300)
    plt.close()
    print("Training curves:", f"training_curves_option_{args.model}_{args.option}_{args.min_len}_{args.max_len}_{args.target_len}_paper_model.png")

def accuracy_euclid(all_ious, args):

    epochs = list(range(1, len(all_ious) + 1))
    
    plt.figure(figsize=(6, 6))
    plt.plot(epochs, all_ious)
    plt.xlabel('Epoch')
    plt.ylabel('Euclidean distance')
    plt.title('Euclidean distance for each epoch')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"accuracy_option_{args.model}_{args.option}_{args.min_len}_{args.max_len}_{args.target_len}_paper_model.png", dpi=300)
    plt.close()
    
def accuracy_iou(all_ious, args):

    epochs = list(range(1, len(all_ious) + 1))
    
    plt.figure(figsize=(6, 6))
    plt.plot(epochs, all_ious)
    plt.xlabel('Epoch')
    plt.ylabel('Intersection over Union')
    plt.title('IoU for each epoch')
    #plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"accuracy_option_{args.model}_{args.option}_{args.min_len}_{args.max_len}_{args.target_len}_paper_model.png", dpi=300)
    plt.close()
    print("Accuracy plot:", f"accuracy_option_{args.model}_{args.option}_{args.min_len}_{args.max_len}_{args.target_len}_paper_model.png")
