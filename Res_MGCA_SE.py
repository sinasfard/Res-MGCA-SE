import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50
from MGCA_SE import *

class Res_MGCA_SE(nn.Module):
    def __init__(self, backbone):
        super(Res_MGCA_SE, self).__init__()

        if backbone == 'res18':
            self.model = resnet18()
        elif backbone == 'res50':
            self.model = resnet50()
        else:
            raise ValueError('Invalid input for network. Use "res18" or "res50".')

    def forward(self, x):
        return self.model(x)
