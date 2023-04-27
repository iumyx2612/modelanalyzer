import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        d_k = Q.size()[-1]
        Kt = K.transpose(1, 2)
        out = torch.matmul(Kt, Q) / math.sqrt(d_k)
        out = F.softmax(out, dim=-1)
        feat_map = torch.matmul(V, out)
        return feat_map