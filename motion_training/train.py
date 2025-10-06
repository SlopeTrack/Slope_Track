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
from loss import bbox_iou, euclid, nll_loss1, nll_loss2, ciou_loss, mse_loss, l1_loss, giou_loss
from plotting import plot_pred, plot_train_val, accuracy_euclid, accuracy_iou
from args import make_parser
from torch.optim.lr_scheduler import LambdaLR
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#global_step=0


def get_teacher_forcing_ratio(step, total_steps, start=0.5, end=0.0):
    """
    Linearly decay teacher forcing ratio from `start` → `end` over training.
    """
    ratio = start - (start - end) * (step / total_steps)
    return max(end, ratio)  # clamp

#global_step=0

def build_scheduler(optimizer, total_epochs, warmup_epochs=200, min_lr=1e-5, base_lr=1e-4):
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
        
        args.teacher_forcing_ratio = get_teacher_forcing_ratio(global_step, total_steps)
        #print(args.teacher_forcing_ratio)
        
        #for t in range(args.target_len):
        if args.nll:
            mean, sigma = model(inp, lengths, args, tgt)
        else:
            mean = model(inp, lengths, args, tgt)
            

        #print( 'Target', tgt.shape, mean.shape)

        # De-normalize to absolute
        #print(mean.shape, tgt.shape)
        mean_abs = denormalize(mean.to(device), ref, last, args, option=opt, delta=delta_tgt, widths=width, heights=height)
        tgt_abs = denormalize(tgt.to(device), ref, last, args, option=opt, delta=delta_tgt, widths=width, heights=height)
        if args.nll:
            #tgt_abs = denormalize(tgt, ref, last, args, option=opt, delta=delta_tgt, widths=width, heights=height)
            sigma_abs = denormalize(sigma.to(device), ref, last, args, option=opt, delta=delta_tgt, widths=width, heights=height)
        #print('Input', inp, 'Mean', mean, 'Mean Abs', mean_abs, 'Target', tgt, 'Target abs', tgt_abs)
        #print('delta',delta_tgt,'sigma',sigma_abs, 'ref', ref)
        #mask = (ref<0).sum().item()
        #print(lengths)
        #print(mask)
        #print( 'Target', tgt.shape, mean.shape)
        
        if args.nll:
            loss_xy = nll_loss2(mean[:,:,:2].to(device), tgt[:,:,:2].to(device), sigma[:,:,:2].to(device)) #, rho[:,:,:2])
            if args.option==5:
                loss = loss_xy
            else:
                #loss_wh = nll_loss2(mean.to(device), tgt.to(device), sigma.to(device)) #, rho[:,:,2:4])
                #nll_loss = loss_wh
                #loss_xy = l1_loss(mean.to(device), tgt)
                #loss_wh = l1_loss(mean[:,:,2:4].to(device), tgt[:,:,2:4])

                loss_xy = l1_loss(mean_abs[:,:,0:2].to(device), tgt_abs[:,:,0:2])
                loss_wh = l1_loss(mean_abs[:,:,2:4].to(device), tgt_abs[:,:,2:4]) 
                l1_loss1 = loss_xy + loss_wh
                loss_xy_nll = nll_loss2(mean[:,:,:2].to(device), tgt[:,:,:2].to(device), sigma[:,:,:2].to(device))
                loss_wh_nll = nll_loss2(mean[:,:,2:4].to(device), tgt[:,:,2:4].to(device), sigma[:,:,2:4].to(device))
                nll_loss = loss_xy_nll + loss_wh_nll
                #print(loss_xy, loss_wh)
        else:
            T = mean.size(1)
            w = torch.linspace(1.0, 2.0, steps=T)
            w = w / w.sum()
            w = w.to(device)
            loss_xy = l1_loss(mean[:,:,0:2].to(device), tgt[:,:,0:2]) #+ nll_loss2(mean[:,:,:2].to(device), tgt[:,:,:2].to(device), sigma[:,:,:2].to(device))
            loss_wh = l1_loss(mean[:,:,2:4].to(device), tgt[:,:,2:4]) #+ nll_loss2(mean[:,:,2:4].to(device), tgt[:,:,2:4].to(device), sigma[:,:,2:4].to(device))
            #loss_y = l1_loss(mean[:,:,1].to(device), tgt[:,:,1]) #+ nll_loss2(mean[:,:,:2].to(device), tgt[:,:,:2].to(device), sigma[:,:,:2].to(device))
            #loss_h = l1_loss(mean[:,:,3].to(device), tgt[:,:,3])
            l1_loss1 = 0.5 * loss_xy + 0.5 * loss_wh   #0.40 * loss_x + 0.3 * loss_w + 0.3 * loss_y + 0.3 * loss_h
            #loss_xy = l1_loss(mean_abs[:,:,0:2].to(device), tgt_abs[:,:,0:2]) #+ nll_loss2(mean[:,:,:2].to(device), tgt[:,:,:2].to(device), sigma[:,:,:2].to(device))
            #l1_loss1 = 0.5 * loss_xy + 0.5 * loss_wh
            #loss_xy = loss_xy.mean(dim=2)
            #loss_wh = loss_wh.mean(dim=2)
            #l1_loss1 = 0.5 * (loss_xy * w).sum(dim=1).mean() + 0.5 * (loss_wh* w).sum(dim=1).mean()
        #mean_abs = denormalize_boxes(mean, width, height)
        #tgt_abs = denormalize_boxes(tgt, width, height)
        if args.option!=5:
            iou = bbox_iou(mean_abs, tgt_abs)
            all_ious.append(iou.detach().cpu().numpy())
            iou_loss = giou_loss(mean_abs, tgt_abs) #(1 - iou)
            #iou_loss = (iou_loss_per * w).sum(dim=1).mean()
            #loss = 0.4 * nll_loss + 0.6 * iou_loss.mean()
            loss = 0.5 * l1_loss1 + 0.5 * iou_loss.mean()
        else: 
            loss = nll_loss
            ious = euclid(mean_abs, tgt_abs)
            #print('Mean_abs', mean_abs, 'Target' , tgt_abs, 'IoUs', ious)
            all_ious.append(ious.detach().cpu().numpy())
            #print(iou_loss.shape) 
        loss.backward()
        optimizer.step()
        total += loss.item()
        global_step+=1
    return total / len(dl), optimizer, np.mean(np.concatenate(all_ious)), global_step 



#@torch.no_grad()
def evaluate(model, dl, opt, args):
    #@torch.no_grad()
    model.eval()
    with torch.no_grad():
        all_ious = []
        total = 0
        for inp, tgt, lengths, ref, last, _,  m, _, _, _, delta_inp, delta_tgt ,_ , _, _, width, height in dl:
            inp, tgt, ref, last, m, delta_inp, delta_tgt = inp.to(device), tgt.to(device), ref.to(device), last.to(device), m.to(device), delta_inp.to(device), delta_tgt.to(device)

            #args.train = False

            args.teacher_forcing_ratio=0.0

            if args.nll:
                mean, sigma = model(inp, lengths, args, tgt)
            else:
                mean = model(inp, lengths, args, tgt)

            #delta = torch.tensor(lengths, device=device).float()
            mean_abs = denormalize(mean.to(device), ref, last, args, option=opt, delta=delta_tgt, widths=width, heights=height)
            tgt_abs = denormalize(tgt.to(device), ref, last, args, option=opt, delta=delta_tgt, widths=width, heights=height)
            if args.nll:
                sigma_abs = denormalize(sigma.to(device), ref, last, args, option=opt, delta=delta_tgt, widths=width, heights=height)

            if args.nll:
                loss_xy = nll_loss2(mean[:,:,:2].to(device), tgt[:,:,:2].to(device), sigma[:,:,:2].to(device))
                if args.option==5:
                    loss = loss_xy
                else:
                    #loss_wh = nll_loss2(mean.to(device), tgt.to(device), sigma.to(device))
                    #print(loss_xy, loss_wh)
                    #nll_loss = loss_wh
                    #loss_xy = l1_loss(mean.to(device), tgt)
                    loss_xy = mse_loss(mean_abs[:,:,0:2].to(device), tgt_abs[:,:,0:2])
                    loss_wh = mse_loss(mean_abs[:,:,2:4].to(device), tgt_abs[:,:,2:4])
                    #loss_wh = l1_loss(mean[:,:,2:4].to(device), tgt[:,:,2:4])
                    l1_loss1 = loss_xy + loss_wh
                    loss_xy_nll = nll_loss2(mean[:,:,:2].to(device), tgt[:,:,:2].to(device), sigma[:,:,:2].to(device))
                    loss_wh_nll = nll_loss2(mean[:,:,2:4].to(device), tgt[:,:,2:4].to(device), sigma[:,:,2:4].to(device))
                    nll_loss = loss_xy_nll + loss_wh_nll
            else:
                #nll_loss = l1_loss(mean.to(device), tgt)
                T = mean.size(1)
                w = torch.linspace(1.0,2.0, steps=T)
                w = w / w.sum()
                w = w.to(device)
                loss_xy = l1_loss(mean[:,:,0:2].to(device), tgt[:,:,0:2]) #+ nll_loss2(mean[:,:,:2].to(device), tgt[:,:,:2].to(device), sigma[:,:,:2].to(device))
                loss_wh = l1_loss(mean[:,:,2:4].to(device), tgt[:,:,2:4]) #+ nll_loss2(mean[:,:,2:4].to(device), tgt[:,:,2:4].to(device), sigma[:,:,2:4].to(device))
                #loss_wh = l1_loss(mean[:,:,2:4].to(device), tgt[:,:,2:4])
                #loss_xy = loss_xy.mean(dim=2)
                #loss_wh = loss_wh.mean(dim=2)
                #l1_loss1 = 0.5 * (loss_xy * w).sum(dim=1).mean() + 0.5 * (loss_wh* w).sum(dim=1).mean()
                l1_loss1 = 0.5 * loss_xy + 0.5 * loss_wh
            #print('NLL',l1_loss1)

            if args.option==5:
                ious = euclid(mean_abs, tgt_abs)
                #print('Mean_abs', mean_abs, 'Target' , tgt_abs, 'IoUs', ious)
                all_ious.append(ious.cpu().numpy())
                loss = nll_loss
            else:
                ious = bbox_iou(mean_abs, tgt_abs)
                #print('Mean_abs', mean_abs, 'Target' , tgt_abs, 'IoUs', ious)
                all_ious.append(ious.cpu().numpy())
                #iou = generalized_box_iou(mean, tgt)
                iou_loss = giou_loss(mean_abs, tgt_abs)  #1 - ious
                #iou_loss = (iou_loss_per * w).sum(dim=1).mean()
                #loss = 0.4 * nll_loss + 0.6 *  iou_loss.mean()
                loss = 0.5 * l1_loss1 + 0.5 * iou_loss.mean() 
                #print('NLL',iou_loss.mean())
            total += loss.item()
        return np.mean(np.concatenate(all_ious)), total / len(dl)

@torch.no_grad()
def predict(model, val_dl, opt, args):
    #@torch.no_grad()
    model.eval()
    val_iter = iter(val_dl)
    for _ in range(1):
        next(val_iter)
    inp, tgt, lengths, ref, last, inp_abs, m, start_idx, end_idx, target_idx, delta_inp, delta_tgt, sequence, old_track_id, frame_num, width, height  = next(val_iter)
    inp, tgt, ref, last, m, delta_inp, delta_tgt = inp.to(device), tgt.to(device), ref.to(device), last.to(device), m.to(device), delta_inp.to(device), delta_tgt.to(device)

    args.train = False

    #print(inp.shape)
    if args.nll:
        mean, sigma = model(inp, lengths, args, tgt)
    else:
        mean = model(inp, lengths, args, tgt)
    #print('Inp',inp, 'Tgt', tgt, 'Mean', mean)
    mean_abs = denormalize(mean, ref, last, args, option=opt, delta=delta_tgt, widths=width, heights=height)
    #print('Target')
    tgt_abs = denormalize(tgt, ref, last, args, option=opt, delta=delta_tgt, widths=width, heights=height)
    print('Delta', delta_inp.shape)
    inp_abs = denormalize_sequence(inp, ref, last, args, option=opt, delta=delta_inp.float().to(device), widths=width, heights=height)
    print('Inp',inp[7], 'Tgt', tgt[7], "Mean", mean[7], 'Inp Abs',inp_abs[7],'Tgt Abs',tgt_abs[7],'Mean Abs', mean_abs[7])
    
    if args.nll:
        return mean_abs, tgt_abs, inp_abs, sigma, start_idx, end_idx, target_idx, frame_num, sequence, old_track_id
    else:
        return mean_abs, tgt_abs, inp_abs, start_idx, end_idx, target_idx, frame_num, sequence, old_track_id
    
def main(args):
    opt = args.option  # choose 1, 2, or 3
    if args.option==5:
       from dataset3 import TrajDataset, collate_fn
       train_ds = TrajDataset("train_set.pkl", option=opt, args=args, augment=True)
       val_ds = TrajDataset("val_set.pkl", option=opt, args=args, augment=False)

    else:
       from dataset3 import TrajDataset, collate_fn

       train_ds = TrajDataset("train_set.pkl", option=opt, args=args, augment=True)
       val_ds = TrajDataset("val_set.pkl", option=opt, args=args, augment=False)
    if args.min_len == 2:
        part_training, _ = torch.utils.data.random_split(train_ds, [0.2, 0.8])
        part_val, _ = torch.utils.data.random_split(val_ds, [0.2, 0.8])
    else:
        part_training, _ = torch.utils.data.random_split(train_ds, [1.0, 0.0])
        part_val, _ = torch.utils.data.random_split(val_ds, [1.0, 0.0])
    
    if args.max_len == args.min_len:
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        print('Loaded training', len(train_dl))
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        print('Loaded val', len(val_dl))
    else:
        train_dl = DataLoader(part_training, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        print('Loaded training', len(train_dl))
        val_dl = DataLoader(part_val, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        print('Loaded val', len(val_dl))

    if args.option==5:
        from lstm import TrajModel
        model = TrajModel().to(device)
    else:
        if args.model == 'mamba':
           from mamba_model import MambaPositionPredictor
           model = MambaPositionPredictor().to(device)
        else:
           from lstm1 import TrajModel
           model = TrajModel(hidden=args.hidden_size).to(device)

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
            #args.teacher_forcing_ratio= get_teacher_forcing_ratio(epoch, args.epochs)
            l, optimizer, train_iou, global_step = train_epoch(model, train_dl, optm, opt, args, global_step,total_steps)
            scheduler.step()
            train_losses.append(l)
            current_lr = scheduler.get_last_lr()[0] #optimizer.param_groups[0]['lr']
            lrs.append(current_lr)
            if epoch % 1 == 0:
                iou, vl = evaluate(model, val_dl, opt, args)
                # scheduler.step()
                val_losses.append(vl)
                all_ious.append(iou)
                print(f"Epoch {epoch:3d}, Train Loss {l:.4f}, Train IoU {train_iou:.4f}, Val loss {vl:.4f}, Val IoU {iou:.4f}", flush=True)
                #print(f"Epoch {epoch:3d}, Train Loss {l:.4f}", flush=True)
        torch.save(model.state_dict(), f"traj_encode_missing_option_{args.model}_{args.option}_{args.min_len}_{args.max_len}_{args.target_len}.pth")

        plot_train_val(train_losses, val_losses, lrs, args)
        if args.option == 5:
            accuracy_euclid(all_ious, args)
        else:
            accuracy_iou(all_ious,args)

    model.load_state_dict(torch.load(f"traj_encode_missing_option_{args.model}_{args.option}_{args.min_len}_{args.max_len}_{args.target_len}_paper_model.pth", weights_only=True))

    if args.nll:
        mean_abs, tgt_abs, inp_abs, sigma, rho, start_idx, end_idx, target_idx, frame_num, sequence, old_track_id = predict(model, val_dl, opt, args)
        #plot_pred(tgt_abs, mean_abs, inp_abs, start_idx, end_idx, target_idx, args, frame_num, sequence, old_track_id, sigma, rho )
        plot_pred(tgt_abs, mean_abs, inp_abs, start_idx, end_idx, target_idx, args, frame_num, sequence, old_track_id)
        #print(inp_abs)
    else:
        mean_abs, tgt_abs, inp_abs, start_idx, end_idx, target_idx, frame_num, sequence, old_track_id = predict(model, val_dl, opt, args)
        plot_pred(tgt_abs, mean_abs, inp_abs, start_idx, end_idx, target_idx, args, frame_num, sequence, old_track_id)

    #print(inp_abs, tgt_abs)

    #plot_pred_nll(tgt_abs, mean_abs, inp_abs, sigma, rho, start_idx, end_idx, target_idx, args, sigma, rho) 

if __name__ == "__main__":
    args = make_parser().parse_args()
    print(args)

    main(args)
