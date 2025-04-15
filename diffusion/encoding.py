import config
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Input x expected shape: [seq_len, batch_size, embedding_dim]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device)
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2: # Handle odd embedding dim
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


# --- PointNet-Style Encoder for Conditioning ---
class PointCloudEncoder(nn.Module):
    def __init__(self, input_dim=config.XY_DIM, embed_dim=config.EMBED_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Shared MLPs implemented using Conv1d
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        # Final MLP after max pooling
        self.mlp2 = nn.Sequential(
            nn.Linear(256, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, point_cloud):
        # point_cloud shape: (B, N_POINTS, XY_DIM)
        x = point_cloud.permute(0, 2, 1) # (B, XY_DIM, N_POINTS)
        point_features = self.mlp1(x) # (B, 256, N_POINTS)
        global_feature, _ = torch.max(point_features, dim=2) # (B, 256)
        condition_embedding = self.mlp2(global_feature) # (B, embed_dim)
        return condition_embedding
