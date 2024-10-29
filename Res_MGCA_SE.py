import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50
from MGCA_SE import *

class Res_MGCA_SE(nn.Module):
    def __init__(self, backbone):
        super(Res_MGCA_SE, self).__init__()

        if backbone == 'res18':
            self.model = resnet18()
            model.layer4[1]=MHCA_resnet_V2(512,64,8,64,512)
            model.layer3[1]=MHCA_resnet_V2(256,64,8,64,256)
            
        elif backbone == 'res50':
            self.model = resnet50()
            model.layer4[1] = MHCA_resnet_V2(2048,128,64,128,2048)
            model.layer4[2] = MHCA_resnet_V2(2048,128,64,128,2048)
            model.layer3[1] = MHCA_resnet_V2(1024,64,64,64,1024)
            model.layer3[2] = MHCA_resnet_V2(1024,64,64,64,1024)
            model.layer3[3] = MHCA_resnet_V2(1024,64,64,64,1024)
            model.layer3[4] = MHCA_resnet_V2(1024,64,64,64,1024)
            model.layer3[5] = MHCA_resnet_V2(1024,64,64,64,1024)
        else:
            raise ValueError('Invalid input for network. Use "res18" or "res50".')

    def forward(self, x):
        return self.model(x)
