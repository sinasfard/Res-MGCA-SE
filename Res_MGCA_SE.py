import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50

class Res_MGCA_SE(nn.Module):
    def __init__(self, backbone, num_classes):
        super(Res_MGCA_SE, self).__init__()

        if backbone == 'res18':
            self.model = resnet18()
            self.model.layer4[1]=MGCA_SE(512,64,8,64,512)
            self.model.layer3[1]=MGCA_SE(256,64,8,64,256)
            self.model.fc = torch.nn.Linear(in_features=512, out_features=num_classes)
            
        elif backbone == 'res50':
            self.model = resnet50()
            self.model.layer4[1] = MGCA_SE(2048,128,64,128,2048)
            self.model.layer4[2] = MGCA_SE(2048,128,64,128,2048)
            self.model.layer3[1] = MGCA_SE(1024,64,64,64,1024)
            self.model.layer3[2] = MGCA_SE(1024,64,64,64,1024)
            self.model.layer3[3] = MGCA_SE(1024,64,64,64,1024)
            self.model.layer3[4] = MGCA_SE(1024,64,64,64,1024)
            self.model.layer3[5] = MGCA_SE(1024,64,64,64,1024)
            self.model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes)
        else:
            raise ValueError('Invalid input for network. Use "res18" or "res50".')

    def forward(self, x):
        return self.model(x)
