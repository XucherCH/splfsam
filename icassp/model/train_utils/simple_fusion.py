import torch
import torch.nn as nn


class FusionModel(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.doubleconv = nn.Sequential(
            nn.Conv2d(2*in_c,out_c, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),

            nn.Conv2d(out_c,out_c, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )

    def forward(self, x1,x2):

        x = torch.cat([x1,x2],dim=1)

        out = self.doubleconv(x)
        return out
    

