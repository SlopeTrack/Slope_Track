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

# ========== Model ==========
class EncoderLSTM(nn.Module):
    def __init__(self, in_dim=2, hidden=256, num_layers=1, bidirectional=False):
        super().__init__()
        self.inlstm = nn.Linear(in_dim, hidden)
        self.lstm = nn.LSTM(in_dim, hidden, num_layers, batch_first=True)

    def forward(self, x, lengths):
        packed = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, hidden_state = self.lstm(packed)
        return hidden_state  #torch.cat((h[-2], h[-1]), dim=1)  h[-1]


class DecoderLSTM(nn.Module):
    def __init__(self, hidden=64, out_dim=2):
        super().__init__()
        self.bilstm = nn.LSTM(out_dim, hidden, 1, batch_first=True)
        self.mlp = nn.Sequential(
            #nn.Dropout(0.5),
            #nn.Linear(hidden, hidden//2),
            #nn.ReLU(),
            #nn.Dropout(0.5),
            #nn.Linear(hidden//2, hidden//4),
            #nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(hidden, out_dim)
            
        )

    def forward(self, x, lengths, h):
        # h: [B, hidden], m: [B]
        h_n, c_n = h
        #x = x[torch.arange(x.shape[0]), torch.tensor(lengths)-1]
        #print('Model output',x)
        output, hidden = self.bilstm(x.unsqueeze(1), (h_n, c_n))
        return output, hidden 
       
class GaussianHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Linear(in_features, 4)

    def forward(self, x):
        out = self.fc(x.squeeze(1))
        """mu = out[:, :4]
        sigma_raw = out[:, 4:6]
        rho_raw = out[:, 6]"""

        mu = out[:, :2]
        sigma_raw = out[:, 2:4]
        #rho_raw = out[:, 4]

        sigma = torch.exp(sigma_raw)           # enforce positive std
        #rho = torch.tanh(rho_raw)              # clamp to [-1, 1]

        return mu, sigma
        

class LinearHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Linear(in_features, 2)

    def forward(self, in_features):
        out = self.fc(in_features.squeeze(1))
        return out
        
class TrajModel(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.enc = EncoderLSTM(hidden=hidden)
        self.dec = DecoderLSTM(hidden=hidden)
        self.gauss_head = GaussianHead(in_features=hidden)
        self.head = LinearHead(in_features=hidden)

    def forward(self, x, lengths, args, target):
        predictions = torch.zeros(x.shape[0], args.target_len, x.shape[2])
        #print(predictions.shape)
        sigma_l = torch.zeros(x.shape[0], args.target_len, x.shape[2])
        #x = x[torch.arange(x.shape[0]), torch.tensor(lengths)-1]
        hidden_state = self.enc(x, lengths)
        x = x[torch.arange(x.shape[0]), torch.tensor(lengths)-1]
        for t in range(args.target_len):
            #hidden_state = self.enc(x, lengths)
            output, hidden_state = self.dec(x, lengths, hidden_state)
            if args.nll:
                output, sigma = self.gauss_head(output)
            else:
                output = self.head(output)
            predictions[:,t,:] = output
            if args.nll:
                sigma_l[:,t,:] = sigma
            if random.random() < args.teacher_forcing_ratio:
                x = target[:, t, :] 
            else:
                x = output
        #print('Predictions shape', predictions.shape)
        if args.nll:
            return predictions, sigma_l
        else:
            return predictions
            
