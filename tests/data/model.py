import torch.nn as nn

from .layers import SelfAttention


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, 10, 3),
            nn.ReLU(),
            nn.Conv2d(10, 10, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.layer_2 = nn.Sequential(
            nn.Conv2d(10, 10, 3),
            nn.ReLU(),
            nn.Conv2d(10, 15, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(1)
        )

    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.classify(out)
        return out


#model CNN combine with attention layer
class CNN_AttentionModel(nn.Module):
    def __init__(self):
        super(CNN_AttentionModel, self).__init__()

        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, 10, 3),
            nn.ReLU(),
            nn.Conv2d(10, 10, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.layer_2 = nn.Sequential(
            nn.Conv2d(10, 10, 3),
            nn.ReLU(),
            nn.Conv2d(10, 15, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer_3 = SelfAttention(15, 10)

        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(1)
        )

    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        B, C, H, W = out.size()
        attention_input = out.permute(0, 2, 3, 1).contiguous().view(B, W * H, C)
        out = self.layer_3(attention_input)
        out = self.classify(out)
        return out