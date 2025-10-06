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


class NormalizedSpatialEncoding(nn.Module):
    def __init__(self, in_dim=4, d_model=128, num_freqs=1):
        """
        in_dim: 4 (dx, dy, dh, dw) normalized
        d_model: output embedding dim
        num_freqs: number of sine/cosine frequencies
        """
        super().__init__()
        self.num_freqs = num_freqs
        self.linear = nn.Linear(in_dim * 2 * num_freqs, d_model)

    def forward(self, norm_inputs):
        """
        norm_inputs: (B, T, 4) already normalized
        Returns: (B, T, d_model)
        """
        B, T, D = norm_inputs.shape
        device = norm_inputs.device

        # frequency bands (geometric progression like in NeRF)
        freqs = torch.arange(self.num_freqs, device=device).float()
        freqs = (10000 ** (-2 * freqs / self.num_freqs)).view(1, 1, 1, -1)

        # expand input with frequencies → (B, T, 4, num_freqs)
        angles = norm_inputs.unsqueeze(-1) / freqs

        # apply sin/cos
        sin_enc = torch.sin(angles)
        cos_enc = torch.cos(angles)

        # concat and flatten → (B, T, 4 * 2 * num_freqs)
        enc = torch.cat([sin_enc, cos_enc], dim=-1).flatten(2)

        # project to model dimension
        return self.linear(enc)

# ========== Model ==========
class EncoderLSTM(nn.Module):
    def __init__(self, in_dim=64, hidden=256, num_layers=2, bidirectional=False):
        super().__init__()
        #self.embedding = nn.Embedding()
        self.inlstm = nn.Linear(in_dim, hidden)
        self.lstm = nn.LSTM(in_dim, hidden, num_layers, batch_first=True, bidirectional=bidirectional, dropout=0.2)
        self.dropout = nn.Dropout(0.2) 

    def forward(self, x, lengths):
        #packed = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, (h_n, c_n) = self.lstm(x)
        h_n = self.dropout(h_n)
        c_n = self.dropout(c_n)
        hidden_state = h_n, c_n
        return output, hidden_state  #torch.cat((h[-2], h[-1]), dim=1)  h[-1]


class DecoderLSTM(nn.Module):
    def __init__(self, hidden=64, out_dim=64):
        super().__init__()
        self.bilstm = nn.LSTM(out_dim, hidden, 2, batch_first=True, bidirectional=False, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        """self.mlp = nn.Sequential(
            #nn.Dropout(0.5),
            #nn.Linear(hidden, hidden//2),
            #nn.ReLU(),
            #nn.Dropout(0.5),
            #nn.Linear(hidden//2, hidden//4),
            #nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(hidden, out_dim)
        )"""

    def forward(self, x, lengths, h):
        # h: [B, hidden], m: [B]
        h_n, c_n = h
        #x = x[torch.arange(x.shape[0]), torch.tensor(lengths)-1]
        #print('Input Shape',x.shape, h_n.shape, c_n.shape)
        output, (h_n, c_n) = self.bilstm(x, (h_n, c_n))
        h_n = self.dropout(h_n)
        c_n = self.dropout(c_n)
        hidden_state = h_n, c_n
        output = self.dropout(output)
        return output, hidden_state


        """def forward(self, x):
        # h: [B, hidden], m: [B]
        #h_n, c_n = h
        #x = x[torch.arange(x.shape[0]), torch.tensor(lengths)-1]
        #print('Input Shape',x.shape, h_n.shape, c_n.shape)
        output, (h_n, c_n) = self.bilstm(x)
        #h_n = self.dropout(h_n)
        #c_n = self.dropout(c_n)
        hidden_state = h_n, c_n
        #output = self.dropout(output)
        return output, hidden_state"""

class GaussianHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Sequential(
            #nn.Dropout(0.1),
            # nn.Linear(in_features, in_features//2),
            #nn.ReLU(),
            #nn.Dropout(0.1),
            #nn.Linear(in_features//2, in_features//4),
            #nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Linear(in_features, 8))
        #self.fc = nn.Linear(in_features, 8)

    def forward(self, x):
        out = self.fc(x.squeeze(1))
        #print(out)
        """mu = out[:, :4]
        sigma_raw = out[:, 4:6]
        rho_raw = out[:, 6]"""

        mu = out[:, :4]
        sigma_raw = out[:, 4:8]
        #rho_raw = out[:, 6]

        sigma = torch.nn.functional.softplus(sigma_raw)          # enforce positive std
        #print(sigma.shape, sigma)
        #rho = torch.tanh(rho_raw)              # clamp to [-1, 1]

        return mu, sigma
        

class LinearHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Sequential(
              #nn.Linear(in_features, in_features//2),
              #nn.ReLU(),
              nn.Linear(in_features, 4))
              #nn.Dropout(0.1))

    def forward(self, in_features):
        out = self.fc(in_features.squeeze(1))
        return out
        
class TrajModel(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.enc = EncoderLSTM(in_dim=hidden, hidden=hidden)
        self.dec = DecoderLSTM(out_dim=hidden, hidden=hidden)
        self.gauss_head = GaussianHead(in_features=hidden)
        self.head = LinearHead(in_features=hidden)
        self.embedding = nn.Embedding(360, 64)
        self.spatial_embedding = NormalizedSpatialEncoding(d_model=hidden)
        self.fuse = nn.Linear(hidden+64, hidden)

    def forward(self, x, lengths, args, target):
        B, T_past, _ = x.shape
        device = x.device

        predictions = torch.zeros(x.shape[0], args.target_len, x.shape[2])
        #print(predictions.shape)
        sigma_l = torch.zeros(x.shape[0], args.target_len, x.shape[2])
        rho = torch.zeros(x.shape[0], args.target_len, x.shape[2])
        t_idx = torch.arange(T_past, device=device).unsqueeze(0).expand(B,-1)
        t_emb = self.embedding(t_idx)
        s_emb = self.spatial_embedding(x)
        #print(t_emb,'S' ,s_emb)
        enc_in = torch.cat([s_emb, t_emb], dim=-1)
        #print(enc_in.shape)
        enc_in = self.fuse(enc_in)
        #print(enc_in)

        output, hidden_state = self.enc(enc_in, lengths)
        x = x[torch.arange(x.shape[0]), torch.tensor(lengths)-1]
        #print(output.shape)
        """h_n, c_n = hidden_state
        output, hidden_state = self.dec(output)
        if args.nll:
           predictions, sigma = self.gauss_head(output)
        else:
           predictions = self.head(output)
        #print(print)"""

        #output, hidden_state = self.dec(x, lengths, hidden_state)
        #output, sigma, rho = self.gauss_head(output)
        for t in range(args.target_len):
            step_idx = T_past + t
            t_emb_step = self.embedding(torch.full((B,), step_idx, device=device))
            #t_emb_step = t_emb_step.unsqueeze(1)
            s_emb = self.spatial_embedding(x.unsqueeze(1)).squeeze(1)
            #print(s_emb.shape, t_emb_step.shape)
            dec_in = torch.cat([s_emb, t_emb_step], dim=-1).unsqueeze(1)
            #print(dec_in.shape)
            dec_in = self.fuse(dec_in)
            
            #dec_in = torch.cat([x.unsqueeze(1), t_emb_step], dim=-1)
            #print('Dec_in',dec_in)
            output, hidden_state = self.dec(dec_in, lengths, hidden_state)
            if args.nll:
                output, sigma = self.gauss_head(output)
            else:
                output = self.head(output)
            predictions[:,t,:] = output
            #print(predictions)
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


    """def forward(self, x, lengths, args, warmup_steps=5):
        batch_size, seq_len, feature_dim = x.shape

        predictions = torch.zeros(batch_size, args.target_len+warmup_steps, feature_dim, device=x.device)
        sigma_l = torch.zeros(batch_size, args.target_len+warmup_steps, feature_dim, device=x.device)
        rho = torch.zeros(batch_size, args.target_len+warmup_steps, feature_dim, device=x.device)

        # Encode past sequence
        hidden_state = self.enc(x, lengths)

        # Start warm-up from last few GT positions
        # Use min(warmup_steps, seq_len) to avoid indexing errors
        warmup_range = min(warmup_steps, seq_len)
        warmup_inputs = x[:, -warmup_range:, :]  # shape: [B, warmup_steps, 4]

        # Step through warm-up
        for t in range(warmup_range):
            decoder_input = warmup_inputs[:, t, :]
            output, hidden_state = self.dec(decoder_input, lengths, hidden_state)
            output, sigma, rho_val = self.gauss_head(output)
            #print('Decoder input', decoder_input, 'Output', output)

            predictions[:, t, :] = output
            sigma_l[:, t, :] = sigma
            #rho[:, t, :] = rho_val

        # Autoregressive prediction from warmup_steps → target_len
        for t in range(warmup_range, args.target_len):
            decoder_input = predictions[:, t - 1, :].detach()  # last predicted step
            output, hidden_state = self.dec(decoder_input, lengths, hidden_state)
            output, sigma, rho = self.gauss_head(output)

            predictions[:, t, :] = output
            sigma_l[:, t, :] = sigma
            #rho[:, t, :] = rho_val

        return predictions, sigma_l, rho"""
