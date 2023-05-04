import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    This is a simulated Self-Attention module that applies on a
    [B, L, C] feature maps
    """
    def __init__(self,
                 input_dim,
                 embed_dims=10):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, embed_dims)
        self.key = nn.Linear(input_dim, embed_dims)
        self.value = nn.Linear(input_dim, embed_dims)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        d_k = Q.size()[-1]
        Kt = K.transpose(1, 2)
        out = torch.matmul(Q, Kt) / math.sqrt(d_k)
        out = F.softmax(out, dim=-1)
        feat_map = torch.matmul(out, V)

        return feat_map
