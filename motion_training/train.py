import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#from mamba_ssm import Mamba2, Mamba
from torchvision.ops import generalized_box_iou
from lstm import TrajModel
#from dataset import TrajDataset, collate_fn
from denorm import denormalize, denormalize_sequence
from loss import bbox_iou, euclid, nll_loss2, ciou_loss, mse_loss, l1_loss, giou_loss
from plotting import plot_pred, plot_train_val, accuracy_euclid, accuracy_iou
from args import make_parser
from torch.optim.lr_scheduler import LambdaLR
import math

random.seed(10)
np.random.seed(10)
torch.manual_seed(10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_scheduler(optimizer, total_epochs, warmup_epochs=100, min_lr=1e-5, base_lr=1e-4):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            # Linear warmup from 0 to base_lr
            return float(current_epoch) / float(max(1, warmup_epochs))
        else:
            # Cosine decay after warmup
            progress = (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return min_lr / base_lr + 0.5 * (1.0 - min_lr / base_lr) * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


# ========== Training ==========
def train_epoch(model, dl, optimizer, opt, args, global_step, total_steps):
    model.train()
    total = 0
    all_ious = []
    for inp, tgt, lengths, ref, last, _,  m, _, _, _, delta_inp, delta_tgt ,_ , _, _, width, height in dl:
        #print(inp, tgt, ref, last, m, delta, width, height)
        inp, tgt, ref, last, m, delta_inp, delta_tgt = inp.to(device), tgt.to(device), ref.to(device), last.to(device), m.to(device), delta_inp.to(device), delta_tgt.to(device)

        optimizer.zero_grad()
        
        mean = model(inp, lengths, args)

        # De-normalize to absolute
        mean_abs = denormalize(mean.to(device), ref, last, args, option=opt, delta=delta_tgt, widths=width, heights=height)
        tgt_abs = denormalize(tgt.to(device), ref, last, args, option=opt, delta=delta_tgt, widths=width, heights=height)

        #Losses
        loss_xy = l1_loss(mean[:,:,0:2].to(device), tgt[:,:,0:2]) 
        loss_wh = l1_loss(mean[:,:,2:4].to(device), tgt[:,:,2:4])
        l1_loss1 = 0.5 * loss_xy + 0.5 * loss_wh   #0.40 * loss_x + 0.3 * loss_w + 0.3 * loss_y + 0.3 * loss_h
        iou_loss = giou_loss(mean_abs, tgt_abs) #(1 - iou)
        loss = 0.5 * l1_loss1 + 0.5 * iou_loss

        # Accuracy
        iou = bbox_iou(mean_abs, tgt_abs)
        all_ious.append(iou.detach().cpu().numpy())
        
        loss.backward()
        optimizer.step()
        total += loss.item()
        global_step+=1
    return total / len(dl), optimizer, np.mean(np.concatenate(all_ious)), global_step 

def evaluate(model, dl, opt, args):
    model.eval()
    with torch.no_grad():
        all_ious = []
        total = 0
        for inp, tgt, lengths, ref, last, _,  m, _, _, _, delta_inp, delta_tgt ,_ , _, _, width, height in dl:
            inp, tgt, ref, last, m, delta_inp, delta_tgt = inp.to(device), tgt.to(device), ref.to(device), last.to(device), m.to(device), delta_inp.to(device), delta_tgt.to(device)
            
            mean = model(inp, lengths, args)

            #Denorm
            mean_abs = denormalize(mean.to(device), ref, last, args, option=opt, delta=delta_tgt, widths=width, heights=height)
            tgt_abs = denormalize(tgt.to(device), ref, last, args, option=opt, delta=delta_tgt, widths=width, heights=height)
            
            #Loss
            loss_xy = l1_loss(mean[:,:,0:2].to(device), tgt[:,:,0:2])
            loss_wh = l1_loss(mean[:,:,2:4].to(device), tgt[:,:,2:4])
            l1_loss1 = 0.5 * loss_xy + 0.5 * loss_wh
            iou_loss = giou_loss(mean_abs, tgt_abs)  #1 - ious
            loss = 0.5 * l1_loss1 + 0.5 * iou_loss
            
            total += loss.item()
            
            ious = bbox_iou(mean_abs, tgt_abs)
            all_ious.append(ious.cpu().numpy())
        return np.mean(np.concatenate(all_ious)), total / len(dl)

@torch.no_grad()
def predict(model, val_dl, opt, args):
    model.eval()
    val_iter = iter(val_dl)
    for _ in range(1):
        next(val_iter)
    inp, tgt, lengths, ref, last, inp_abs, m, start_idx, end_idx, target_idx, delta_inp, delta_tgt, sequence, old_track_id, frame_num, width, height  = next(val_iter)
    inp, tgt, ref, last, m, delta_inp, delta_tgt = inp.to(device), tgt.to(device), ref.to(device), last.to(device), m.to(device), delta_inp.to(device), delta_tgt.to(device)

    args.train = False

    mean = model(inp, lengths, args)
    
    mean_abs = denormalize(mean, ref, last, args, option=opt, delta=delta_tgt, widths=width, heights=height)
    tgt_abs = denormalize(tgt, ref, last, args, option=opt, delta=delta_tgt, widths=width, heights=height)
    inp_abs = denormalize_sequence(inp, ref, last, args, option=opt, delta=delta_inp.float().to(device), widths=width, heights=height)
    
    print('Inp',inp[7], 'Tgt', tgt[7], "Mean", mean[7], 'Inp Abs',inp_abs[7],'Tgt Abs',tgt_abs[7],'Mean Abs', mean_abs[7])

    return mean_abs, tgt_abs, inp_abs, start_idx, end_idx, target_idx, frame_num, sequence, old_track_id
    
def main(args):
    opt = args.option 
    from dataset3 import TrajDataset, collate_fn
    train_ds = TrajDataset("train_set.pkl", option=opt, args=args, augment=True)
    val_ds = TrajDataset("val_set.pkl", option=opt, args=args, augment=False)

    part_training, _ = torch.utils.data.random_split(train_ds, [1.0, 0.0])
    part_val, _ = torch.utils.data.random_split(val_ds, [1.0, 0.0])
    
    g = torch.Generator()
    g.manual_seed(10)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, generator=g)
    print('Loaded training', len(train_dl))
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    print('Loaded val', len(val_dl))

    from mamba_model import MambaPositionPredictor
    model = MambaPositionPredictor().to(device)
    
    optm = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = build_scheduler(optm, args.epochs)

    if args.train:
        train_losses = []
        val_losses = []
        lrs = []
        all_ious = []
        total_steps = len(train_dl)*args.epochs+1
        global_step=0
        for epoch in range(1, args.epochs+1):
            l, optimizer, train_iou, global_step = train_epoch(model, train_dl, optm, opt, args, global_step,total_steps)
            scheduler.step()
            train_losses.append(l)
            current_lr = optimizer.param_groups[0]['lr']
            lrs.append(current_lr)
            if epoch % 1 == 0:
                iou, vl = evaluate(model, val_dl, opt, args)
                val_losses.append(vl)
                all_ious.append(iou)
                print(f"Epoch {epoch:3d}, Train Loss {l:.4f}, Train IoU {train_iou:.4f}, Val loss {vl:.4f}, Val IoU {iou:.4f}", flush=True)
        model_name=f"traj_encode_missing_option_{args.model}_{args.option}_{args.min_len}_{args.max_len}_{args.target_len}_dropout_2_3_2_0_2_1_2_x_semb_8_synthetic_{args.use_synthetic}_03_sampled_augment_norm.pth"
        torch.save(model.state_dict(), model_name)

        print("Model saved at:", model_name)
        
        #Training plots
        plot_train_val(train_losses, val_losses, lrs, args)
        accuracy_iou(all_ious,args)

    #Test
    model.load_state_dict(torch.load(model_name, weights_only=True))

    mean_abs, tgt_abs, inp_abs, start_idx, end_idx, target_idx, frame_num, sequence, old_track_id = predict(model, val_dl, opt, args)
    plot_pred(tgt_abs, mean_abs, inp_abs, start_idx, end_idx, target_idx, args, frame_num, sequence, old_track_id)
 

if __name__ == "__main__":
    args = make_parser().parse_args()
    print(args)
    main(args)
