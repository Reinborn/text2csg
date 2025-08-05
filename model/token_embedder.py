
import torch
import torch.nn as nn

class CSGTokenEmbedder(nn.Module):
    def __init__(self,
                 type_vocab_size=6,
                 surf_type_vocab_size=3,
                 sign_vocab_size=2,
                 axis_vocab_size=3,
                 coeff_vocab_size=256,
                 surface_vocab_size=1000,
                 parent_vocab_size=1000,
                 depth_vocab_size=20,
                 emb_dim=32):
        super().__init__()
        self.type_emb = nn.Embedding(type_vocab_size, emb_dim)
        self.surf_type_emb = nn.Embedding(surf_type_vocab_size, emb_dim)
        self.sign_emb = nn.Embedding(sign_vocab_size, emb_dim)
        self.axis_emb = nn.Embedding(axis_vocab_size, emb_dim)
        self.coeff_emb = nn.Embedding(coeff_vocab_size, emb_dim)
        self.surface_emb = nn.Embedding(surface_vocab_size, emb_dim)
        self.parent_emb = nn.Embedding(parent_vocab_size, emb_dim)
        self.depth_emb = nn.Embedding(depth_vocab_size, emb_dim)

        self.output_proj = nn.Linear(emb_dim * 8, emb_dim)

    def forward(self, tokens):
        t0 = self.type_emb(tokens[..., 0])
        t1 = self.surf_type_emb(tokens[..., 1])
        t2 = self.sign_emb(tokens[..., 2])
        t3 = self.axis_emb(tokens[..., 3])
        t4 = self.coeff_emb(tokens[..., 4])
        t5 = self.surface_emb(tokens[..., 5])
        t6 = self.parent_emb(tokens[..., 6])
        t7 = self.depth_emb(tokens[..., 7])
        token_feature = torch.cat([t0, t1, t2, t3, t4, t5, t6, t7], dim=-1)
        return self.output_proj(token_feature)
