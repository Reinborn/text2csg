
import torch
import torch.nn as nn
from model.token_embedder import CSGTokenEmbedder
from model.tree_guided_attention import TreeGuidedAttention
import torch.nn.functional as F

class TreeAwareTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.tree_attn = TreeGuidedAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, tgt_mask, parent_mask, sibling_mask):
        self_attn_output = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, need_weights=False)[0]
        tgt = self.norm1(tgt + self_attn_output)

        tree_output = self.tree_attn(tgt, parent_mask, sibling_mask)
        tgt = self.norm2(tgt + tree_output)

        ff = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout(ff))
        return tgt

class CSGTreeDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedder = CSGTokenEmbedder(
            surface_vocab_size=config.max_surface_id,
            parent_vocab_size=config.max_parent_id,
            depth_vocab_size=config.max_depth,
            emb_dim=config.embed_dim
        )

        self.layers = nn.ModuleList([
            TreeAwareTransformerDecoderLayer(
                d_model=config.embed_dim,
                nhead=config.nhead,
                dim_feedforward=config.ff_dim,
                dropout=config.dropout
            ) for _ in range(config.num_layers)
        ])

        self.output_head = nn.Linear(config.embed_dim, config.output_dim)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)

    def forward(self, tokens, parent_mask, sibling_mask):
        x = self.embedder(tokens)  # (B, L, D)
        B, L, D = x.shape
        tgt_mask = self.generate_square_subsequent_mask(L).to(x.device)

        for layer in self.layers:
            x = layer(x, tgt_mask, parent_mask, sibling_mask)

        return self.output_head(x)
