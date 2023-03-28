import torch.nn as nn


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