import torch
import torch.nn as nn
from mamba_ssm import Mamba
import torch.nn.functional as F

torch.manual_seed(10)

class SpatialPosEncoding(nn.Module):
    def __init__(self, in_dim=4, d_model=128, num_freqs=8):
        super().__init__()
        self.num_freqs = num_freqs
        self.linear = nn.Linear(in_dim * 2 * num_freqs, d_model)

    def forward(self, bboxes, lengths):
        """
        Args:
            bboxes: (B, T, 4) normalized bounding boxes
            lengths: (B,) number of valid timesteps per batch
        Returns:
            (B, T, d_model)
        """
        B, T, D = bboxes.shape
        device = bboxes.device

        lengths_tensor = torch.tensor(lengths, device=device)
        # Create mask: True for valid positions, False for padding
        mask = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)  # (B, T)
        mask = mask < lengths_tensor.unsqueeze(1)  # (B, T)

        # Frequencies
        freqs = torch.arange(self.num_freqs, device=device).float()
        freqs = (10000 ** (-2 * freqs / self.num_freqs)).view(1, 1, 1, -1)

        # Expand bboxes -> (B, T, 4, num_freqs)
        angles = bboxes.unsqueeze(-1) / freqs
        sin_enc = torch.sin(angles)
        cos_enc = torch.cos(angles)

        # Concat sin+cos -> (B, T, 4, 2*num_freqs)
        enc = torch.cat([sin_enc, cos_enc], dim=-1)

        # Flatten -> (B, T, 4*2*num_freqs)
        enc = enc.flatten(2)

        # Zero out padded positions
        mask = mask.unsqueeze(-1)  # (B, T, 1)
        enc = enc * mask.float()

        # Linear projection
        return self.linear(enc)

class LinearHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc_xy = nn.Sequential(
                  nn.Dropout(0.2),#0.2
                  nn.Linear(in_features, in_features//2), 
                  nn.Tanh(),
                  nn.Linear(in_features//2, in_features//4),
                  nn.ReLU(),
                  nn.Linear(in_features//4, 2),
                  )
        
        self.fc_wh = nn.Sequential(
                  nn.Dropout(0.3),
                  nn.Linear(in_features, in_features//2), 
                  nn.Tanh(),
                  nn.Linear(in_features//2, in_features//4),
                  nn.ReLU(),
                  nn.Linear(in_features//4, 2)) 

    def forward(self, in_features):
        xy = self.fc_xy(in_features)
        wh = self.fc_wh(in_features)
        out = torch.cat([xy,wh], dim =-1)
        return out

class MambaPositionPredictor(nn.Module):
    def __init__(self, in_dim=36, d_model=96, d_state=8, d_conv=4, expand=2,
                 num_layers=1, hidden=96):
        super().__init__()
        
        self.in_proj = nn.Linear(in_dim, d_model)

        # === Mamba encoder stack ===
        self.mamba_blocks = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])
                     
        self.mamba_blocks2 = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])
        self.embedding = nn.Embedding(360,32)

        self.spatial_embedding = SpatialPosEncoding(d_model=32)
       
        self.norm1 = nn.LayerNorm(d_model*3)
       
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout4 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.1)
                     
        self.head = LinearHead(in_features=hidden*3)

    def forward(self, x, lengths, args):
            """
            Args:
            x: (B, T_in, in_dim)
            lengths: int, valid positions
            args
            Returns:
            preds: (B, target_len, in_dim)
            """
            B, T_past, _ = x.shape
            device = x.device
            lengths_tensor = torch.tensor(lengths, device=device)  # (B,)
            s_emb = self.spatial_embedding(x, lengths)
            enc_in = torch.cat([x, s_emb], dim=-1)
            new_enc_in = self.in_proj(new_enc_in)
        
            # === Encode history with Mamba ===
            h = self.dropout1(new_enc_in)  # (B, T_in, d_model)
            for block in self.mamba_blocks:
                h = block(h)
                h = self.dropout2(h)

            h_bwd = torch.flip(new_enc_in, dims=[1])  # (B, T_in, d_model)
            for block in self.mamba_blocks2:
                h_bwd = block(h_bwd)
                h_bwd = self.dropout3(h_bwd)
            h_bwd = torch.flip(h_bwd, dims=[1])
        
            output = self.norm1(torch.cat([h, h_bwd, self.dropout4(new_enc_in)], dim= -1))

            mu_all = self.head(output)
            return mu_all[:,:args.target_len,:]

