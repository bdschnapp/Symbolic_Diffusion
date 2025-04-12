from diffusion.encoding import *
import torch.nn as nn
import torch


class ConditionalD3PMTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, dim_feedforward,
                 seq_len, condition_dim,
                 num_timesteps, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.num_layers = num_layers

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=config.PAD_TOKEN_ID)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout, max_len=seq_len + 1)
        self.timestep_embedding = nn.Sequential(
            TimestepEmbedding(embed_dim),
            nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )
        # Use PointCloudEncoder for condition
        self.condition_encoder = PointCloudEncoder(input_dim=config.XY_DIM, embed_dim=embed_dim)

        # Optional: Condition Dropout Probability
        self.condition_dropout_prob = 0.1 # Set to 0 to disable

        # Manual Transformer Block Components
        self.encoder_self_attn_layers = nn.ModuleList()
        self.encoder_cross_attn_layers = nn.ModuleList()
        self.encoder_ffn_layers = nn.ModuleList()
        self.encoder_norm1_layers = nn.ModuleList()
        self.encoder_norm2_layers = nn.ModuleList()
        self.encoder_norm3_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.encoder_self_attn_layers.append(nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True))
            self.encoder_cross_attn_layers.append(nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True))
            self.encoder_ffn_layers.append(nn.Sequential(
                nn.Linear(embed_dim, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(dim_feedforward, embed_dim)
            ))
            self.encoder_norm1_layers.append(nn.LayerNorm(embed_dim))
            self.encoder_norm2_layers.append(nn.LayerNorm(embed_dim))
            self.encoder_norm3_layers.append(nn.LayerNorm(embed_dim))
            self.dropout_layers.append(nn.Dropout(dropout))

        self.output_layer = nn.Linear(embed_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        if self.token_embedding.padding_idx is not None:
             self.token_embedding.weight.data[self.token_embedding.padding_idx].zero_()
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)
        # Init PointCloudEncoder layers
        for layer in self.condition_encoder.modules():
             if isinstance(layer, (nn.Conv1d, nn.Linear)):
                 layer.weight.data.normal_(mean=0.0, std=0.02)
                 if layer.bias is not None:
                     layer.bias.data.zero_()
             elif isinstance(layer, nn.BatchNorm1d):
                 layer.weight.data.fill_(1.0)
                 layer.bias.data.zero_()

    def forward(self, x, t, condition):
        # CONDITION INPUT SHAPE CHANGE: Expects (B, N_POINTS, XY_DIM)
        batch_size, seq_len = x.shape
        device = x.device

        # 1. Embeddings
        token_emb = self.token_embedding(x) * math.sqrt(self.embed_dim)
        token_emb_permuted = token_emb.transpose(0, 1)
        pos_emb_permuted = self.positional_encoding(token_emb_permuted)
        pos_emb = pos_emb_permuted.transpose(0, 1)
        time_emb = self.timestep_embedding(t)
        time_emb = time_emb.unsqueeze(1).expand(-1, seq_len, -1)

        # 2. Condition Embedding
        cond_emb_proj = self.condition_encoder(condition)

        # Optional: Condition Dropout
        if self.training and self.condition_dropout_prob > 0:
            mask = (torch.rand(cond_emb_proj.shape[0], 1, device=cond_emb_proj.device) > self.condition_dropout_prob).float()
            cond_emb_proj = cond_emb_proj * mask

        cond_kv = cond_emb_proj.unsqueeze(1)

        # 3. Initial sequence representation
        current_input = pos_emb + time_emb

        # 4. Padding mask
        padding_mask = (x == config.PAD_TOKEN_ID)

        for i in range(self.num_layers):
            # Self-Attention
            sa_norm_input = self.encoder_norm1_layers[i](current_input)
            sa_output, _ = self.encoder_self_attn_layers[i](query=sa_norm_input, key=sa_norm_input, value=sa_norm_input, key_padding_mask=padding_mask)
            x = current_input + self.dropout_layers[i](sa_output)
            # Cross-Attention
            ca_norm_input = self.encoder_norm3_layers[i](x)
            ca_output, _ = self.encoder_cross_attn_layers[i](query=ca_norm_input, key=cond_kv, value=cond_kv)
            x = x + self.dropout_layers[i](ca_output)
            # Feed-Forward
            ffn_norm_input = self.encoder_norm2_layers[i](x)
            ffn_output = self.encoder_ffn_layers[i](ffn_norm_input)
            x = x + ffn_output
            current_input = x

        transformer_output = current_input
        output_logits = self.output_layer(transformer_output)
        return output_logits
