
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TreeGuidedAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, parent_mask, sibling_mask):
        B, L, D = x.shape
        H = self.nhead

        Q = self.q_proj(x).view(B, L, H, -1).transpose(1, 2)  # (B, H, L, d)
        K = self.k_proj(x).view(B, L, H, -1).transpose(1, 2)
        V = self.v_proj(x).view(B, L, H, -1).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, L, L)
        mask = (parent_mask | sibling_mask).unsqueeze(1)  # (B, 1, L, L)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # (B, H, L, d)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(attn_output)

class TreeAwareTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.tree_attn = TreeGuidedAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, parent_mask, sibling_mask):
        # standard self-attention
        src2 = self.self_attn(src, src, src, need_weights=False)[0]
        src = self.norm1(src + self.dropout1(src2))

        # tree-guided attention
        guided = self.tree_attn(src, parent_mask, sibling_mask)
        src = self.norm2(src + self.dropout2(guided))

        # feedforward
        ff = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm3(src + self.dropout(ff))
        return src
