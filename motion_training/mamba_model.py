import torch
import torch.nn as nn
from mamba_ssm import Mamba
import torch.nn.functional as F

torch.manual_seed(10)

"""class TemporalSpatialFusion(nn.Module):
    def __init__(self, hidden_dim=64, time_dim=32, n_heads=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim

        # Project time embedding into hidden_dim
        self.time_proj = nn.Linear(time_dim, hidden_dim)

        # Multi-head attention for fusion
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                          num_heads=n_heads,
                                          dropout=dropout,
                                          batch_first=True)

        # Residual + feedforward block
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, s_emb, t_emb):
        
        Args:
            s_emb: (B, T, H) spatial embeddings
            t_emb: (B, T, D_t) temporal embeddings
        Returns:
            fused: (B, T, H)
        
        # Project temporal into hidden_dim
        t_emb_proj = self.time_proj(t_emb)  # (B, T, H)

        # Treat spatial as queries, temporal as keys/values
        fused, _ = self.attn(query=s_emb, key=t_emb_proj, value=t_emb_proj)

        # Residual connection + norm
        fused = self.norm1(s_emb + fused)

        # Feedforward
        fused = fused + self.ff(fused)
        fused = self.norm2(fused)

        return fused  # (B, T, H)



class BiDirectionalFusion(nn.Module):
    def __init__(self, hidden_dim=64, time_dim=32, n_heads=2, dropout=0.1):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, hidden_dim)

        self.attn_s2t = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.attn_t2s = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, s_emb, t_emb):
        # project time into hidden space
        t_emb_proj = self.time_proj(t_emb)

        # Spatial → Temporal
        s2t, _ = self.attn_s2t(query=s_emb, key=t_emb_proj, value=t_emb_proj)

        # Temporal → Spatial
        t2s, _ = self.attn_t2s(query=t_emb_proj, key=s_emb, value=s_emb)

        # Combine (residual connections)
        fused = s_emb + s2t + t2s
        fused = self.norm(fused)
        return fused  # (B, T, H)

class SelfAttention(nn.Module):

    def __init__(self, dim, num_heads=2, dropout=0.3):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, N, D = x.shape  # batch, seq_len, embed_dim
        qkv = self.qkv(x)                        # (B, N, 3*D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)              # each (B, N, H, Hd)
        q = q.transpose(1, 2)                    # (B, H, N, Hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B,H,N,N)

        if mask is not None:                     # optional mask (e.g. causal for t2t)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v)                          # (B,H,N,Hd)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        return out"""

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


"""class NormalizedSpatialEncoding(nn.Module):
    def __init__(self, in_dim=4, d_model=128, num_freqs=4):
        
        in_dim: 4 (dx, dy, dh, dw) normalized
        d_model: output embedding dim
        num_freqs: number of sine/cosine frequencies
        
        super().__init__()
        self.num_freqs = num_freqs
        self.linear = nn.Linear(in_dim * 2 * num_freqs, d_model)

    def forward(self, norm_inputs):
        
        norm_inputs: (B, T, 4) already normalized
        Returns: (B, T, d_model)
        
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
        return self.linear(enc)"""


class LinearHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc_xy = nn.Sequential(
                  nn.Dropout(0.2),#0.2
                  nn.Linear(in_features, in_features//2), 
                  #nn.Dropout(0.3),
                  nn.Tanh(),
                  #nn.Dropout(0.2),
                  nn.Linear(in_features//2, in_features//4),
                  #nn.Tanh(),
                  nn.ReLU(),
                  #nn.Dropout(0.2),
                  #nn.Linear(in_features//4, in_features//8),
                  #nn.Tanh(),
                  #nn.Dropout(0.3),
                  nn.Linear(in_features//4, 2),
                  )
        self.fc_wh = nn.Sequential(
                  nn.Dropout(0.3),
                  nn.Linear(in_features, in_features//2), 
                  nn.Tanh(),
                  #nn.Dropout(0.2),
                  nn.Linear(in_features//2, in_features//4),
                  #nn.Tanh(),
                  nn.ReLU(),
                  #nn.Dropout(0.2),
                  #nn.Linear(in_features//4, in_features//8),
                  #nn.Tanh(),
                  #nn.Dropout(0.3),
                  nn.Linear(in_features//4, 2)) 
        #self.fc = nn.Linear(in_features, 4)

    def forward(self, in_features):
        xy = self.fc_xy(in_features)
        wh = self.fc_wh(in_features)
        out = torch.cat([xy,wh], dim =-1)
        return out



"""class GaussianHead(nn.Module):
    def __init__(self, in_features, out_dim=4):
        super().__init__()
        self.out_dim = out_dim
        
        #self.fc = nn.Linear(in_features, out_dim * 2)  # mu + sigma
        self.fc_xy = nn.Sequential(
                  nn.Dropout(0.3),
                  nn.Linear(in_features, in_features//2), 
                  #nn.Dropout(0.3),
                  nn.Tanh(),
                  #nn.Dropout(0.2),
                  nn.Linear(in_features//2, in_features//4),
                  nn.ReLU(),
                  nn.Linear(in_features//4, in_features//8),
                  nn.Tanh(),
                  nn.Linear(in_features//8, 4)
                  )
        self.fc_wh = nn.Sequential(
                  nn.Dropout(0.3),
                  nn.Linear(in_features, in_features//2), 
                  nn.Tanh(),
                  #nn.Dropout(0.2),
                  nn.Linear(in_features//2, 4))
        
        
    def forward(self, in_features):
        
        #out = self.fc(x)
        xy = self.fc_xy(in_features)
        wh = self.fc_wh(in_features)
        out = torch.cat([xy,wh], dim =-1)
        #out = self.out_proj(x)         # (B, pred_len * d_model)
        #out = out.view(-1, args.target_len, self.out_dim * 2)
        mu = out[:,:, :4]
        sigma_raw = out[:, :, 4:8]
        #rho_raw = out[:, 6:8]

        sigma = torch.nn.functional.softplus(sigma_raw)+ 1e-3  #torch.exp(sigma_raw)   # enforce positivity
        #sigma = sigma
        #rho = torch.tanh(rho_raw)      # clamp to [-1, 1]

        return mu, sigma"""

"""class DecoderLSTM(nn.Module):
    def __init__(self, hidden=64, out_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(out_dim, hidden, 1, batch_first=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)   # output: (B, 1, hidden)
        #pred = self.mlp(output.squeeze(1)      # (B, out_dim)
        h_n, c_n = hidden
        h_n = self.dropout(h_n)
        c_n = self.dropout(c_n)
        hidden= h_n, c_n
        return self.dropout(output), hidden"""


class MambaPositionPredictor(nn.Module):
    def __init__(self, in_dim=36, d_model=96, d_state=8, d_conv=4, expand=2,
                 num_layers=1, hidden=96, pred_len=10):
        super().__init__()
        #self.pred_len = pred_len
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

        self.embedding = nn.Embedding(360, 32)
        self.spatial_embedding = SpatialPosEncoding(d_model=32)
        #self.fuse = nn.Sequential(nn.Linear(179+179, d_model), nn.LayerNorm(d_model))
        """self.fuse = TemporalSpatialFusion(hidden_dim=64,
                                  time_dim=64,
                                  n_heads=4,
                                  dropout=0.3)"""
        #self.fuse = BiDirectionalFusion(hidden_dim=32, time_dim=32)
        #self.num_layers = num_layers
        #self.activation= nn.ReLU()
        self.norm1 = nn.LayerNorm(d_model*3)
        self.norm2 = nn.LayerNorm(in_dim)
        #self.norm3 = nn.LayerNorm(d_model)
        #d_ff = 1 * d_model
        #self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        #self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
       

        #self.proj = nn.Linear(192, d_model)
        #self.final_proj = nn.Linear(192, d_model) 
        # === Project Mamba hidden to LSTM init ===
        #self.h_proj = nn.Sequential(nn.Linear(d_model, hidden),nn.Tanh(), nn.Dropout(0.3), nn.LayerNorm(hidden))
        #self.c_proj = nn.Sequential(nn.Linear(d_model, hidden),nn.Tanh(), nn.Dropout(0.3), nn.LayerNorm(hidden))

        #self.c_proj = nn.Sequential(nn.Linear(d_model, hidden))
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout4 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.1)

        # === LSTM decoder ===
        #self.decoder = DecoderLSTM(hidden=hidden)
        #self.gauss_head = GaussianHead(in_features=hidden*3)
        self.head = LinearHead(in_features=hidden*3)

    def forward(self, x, lengths, args, target):
            """
            Args:
            x: (B, T_in, in_dim)
            target_len: int, horizon length
            teacher_forcing: float in [0,1], prob of using GT during training
            targets: (B, target_len, in_dim) if teacher forcing enabled
            Returns:
            preds: (B, target_len, in_dim)
            """
            #B, T_past, _ = x.shape
            #device = x.device

            #mu_all, sigma_all, rho_all = [], [], []
            #for t in range(args.target_len):
            B, T_past, _ = x.shape
            device = x.device
            #t_idx = torch.arange(T_past, device=device).unsqueeze(0).expand(B,-1)
            #t_emb = self.embedding(t_idx)
            # lengths: list or tensor of shape (B,)
            #x = torch.abs(x)
            lengths_tensor = torch.tensor(lengths, device=device)  # (B,)

            # Create a mask of valid steps: (B, T)
            valid_mask = torch.arange(T_past, device=device).unsqueeze(0) < lengths_tensor.unsqueeze(1)

            # Lookup embeddings as before
            t_idx = torch.arange(T_past, device=device).unsqueeze(0).expand(B, -1)
            t_emb_all = self.embedding(t_idx)  # (B, T, time_emb_dim)

            # Zero out embeddings for padded steps
            t_emb = t_emb_all * valid_mask.unsqueeze(-1)

            s_emb = self.spatial_embedding(x, lengths)
            #mask_expanded = valid_mask.unsqueeze(-1).float()
            #x = x * mask_expanded
            enc_in = torch.cat([x, s_emb], dim=-1)
            #enc_in = s_emb + t_emb + self.in_proj(x)
            #t_emb = self.time_attention(t_emb)
            #s_emb = self.space_attention(s_emb)
            new_enc_in = enc_in
            #new_enc_in = self.time_attention(enc_in)
            #new_enc_in = enc_in #+ s_emb #+ t_emb  
            #print(enc_in.shape)
            #enc_in = self.fuse(s_emb, t_emb)


            # === Encode history with Mamba ===
            h = self.dropout1(self.in_proj(new_enc_in))  # (B, T_in, d_model)
            #h = h_original
            for block in self.mamba_blocks:
                #h = block(h)
                """if len(self.mamba_blocks)>1 and i == 0:
                    h_out = block(h)
                    h = self.norm2(self.dropout2(self.final_proj(torch.cat([h_out, h], dim=-1))))
                else:"""
                h = block(h)
                h = self.dropout2(h)
                #h = self.dropout(h + h_original)
                #h = self.dropout(self.proj(torch.cat([h ,h_original], dim=-1)))

            h_bwd = torch.flip(self.in_proj(new_enc_in), dims=[1])  # (B, T_in, d_model)
            #h_bwd = h_bwd_original
            for block in self.mamba_blocks2:
                """if len(self.mamba_blocks)>1 and i == 0:
                    h_bwd_out = block(h_bwd)
                    h_bwd = self.norm2(self.dropout2(self.final_proj(torch.cat([h_bwd_out,h_bwd], dim=-1))))
                else:"""
                h_bwd = block(h_bwd)
                h_bwd = self.dropout3(h_bwd)
                #h_bwd = self.dropout(h_bwd + h_bwd_original)
                #    h_bwd = self.dropout2(self.norm2(self.proj(torch.cat([h_bwd, h_bwd_original], dim=-1))))
                #h_bwd = h_bwd.permute(0,2,1)
                """else:
                #h_bwd = self.dropout2(h_bwd)
                h_bwd_out = block(h_bwd + h_bwd_original)
                h_bwd = self.dropout(h_bwd_out + h_bwd"""
            h_bwd = torch.flip(h_bwd, dims=[1])
            output = self.norm1(torch.cat([h, h_bwd, self.dropout4(self.in_proj(enc_in))], dim= -1))
            """if args.nll:
                mu, sigma = self.gauss_head(output)
                sigma_all.append(sigma)
            else:
                mu = self.head(output)
            #print(prev.unsqueeze(1).shape, h_0.shape, c_0.shape, mu.shape )
            #preds.append(pred)
            
            mu_all.append(mu[:,0,:])
            #sigma_all.append(sigma)
            #rho_all.append(rho)

            # Teacher forcing if enabled
            if (args.train and args.teacher_forcing_ratio is not None 
                and target is not None 
                and torch.rand(1).item() < args.teacher_forcing_ratio):
                prev = target[:, :t, :]
            else:
                prev = mu[:,:1,:]

            x = torch.cat([x, prev], dim=1)

            mu_all = torch.stack(mu_all, dim=1)       # (B, pred_len, in_dim)
            #print(mu_all.shape)
            if args.nll:
            sigma_all = torch.stack(sigma_all, dim=1) # (B, pred_len, in_dim)
            rho_all = torch.stack(rho_all, dim=1)
            #preds = torch.stack(preds, dim=1)  # (B, target_len, 4)
            return mu_all, sigma_all
            else:
            return mu_all"""

            if args.nll:
               mu_all, sigma_all = self.gauss_head(output)
               return mu_all[:,:args.target_len,:], sigma_all[:,:args.target_len,:]
            #sigma_all.append(sigma)
            else:
               mu_all = self.head(output)
               return mu_all[:,:args.target_len,:]
            #print(mu_all)
            #print(prev.unsqueeze(1).shape, h_0.shape, c_0.shape )

            """for t in range(args.target_len):
            #print(prev.shape, h[:, -1, :].shape, h_bwd[:, -1, :].shape)
            output = self.norm1(torch.cat([prev, h[:, -1, :], h_bwd[:, -1, :]], dim= -1))
            #output, hidden = self.decoder(dec_in, hidden)  # (B,4)
            h = self.final_proj(output.unsqueeze(1))
            for block in self.mamba_blocks2:
                h = block(h)
                h = self.dropout2(h)

            h_bwd_original = torch.flip(self.final_proj(output.unsqueeze(1)), dims=[1])  # (B, T_in, d_model)
            h_bwd = h_bwd_original
            for block in self.mamba_blocks2:
                h_bwd = block(h_bwd)
                #if self.num_layers>1:
                #    h_bwd = self.dropout2(self.norm2(self.proj(torch.cat([h_bwd, h_bwd_original], dim=-1))))
                #h_bwd = h_bwd.permute(0,2,1)
                #else:
                h_bwd = self.dropout2(h_bwd)
            if args.nll:
                mu, sigma = self.gauss_head(output)
                sigma_all.append(sigma)
            else:
                mu = self.head(torch.cat([h, h_bwd], dim=-1))
            #print(prev.unsqueeze(1).shape, h_0.shape, c_0.shape, mu.shape )
            #preds.append(pred)
            
            mu_all.append(mu)
            #sigma_all.append(sigma)
            #rho_all.append(rho)

            # Teacher forcing if enabled
            if (args.train and args.teacher_forcing_ratio is not None 
                and target is not None 
                and torch.rand(1).item() < args.teacher_forcing_ratio):
                prev = target[:, t, :]
            else:
                prev = mu.squeeze(1)

            mu_all = torch.stack(mu_all, dim=1)       # (B, pred_len, in_dim)
            #print(mu_all.shape)
            if args.nll:
            sigma_all = torch.stack(sigma_all, dim=1) # (B, pred_len, in_dim)
            rho_all = torch.stack(rho_all, dim=1)
            #preds = torch.stack(preds, dim=1)  # (B, target_len, 4)
            return mu_all, sigma_all
            else:
            return mu_all.squeeze(2)"""

