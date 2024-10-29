import torch
import torch.nn as nn
from SE_block import SEBlock

class MGCA_SE(nn.Module):
    """
    Multi-Head Convolutional Attention
    """
    def __init__(self, in_channels, out_channels, head_dim, projection_out_channels, channels_next_layer):
        super(MHCA_resnet_V2, self).__init__()

        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)

        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv1x1_3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)


        self.group_conv3x3 = nn.Conv2d(out_channels*3, out_channels, kernel_size=3, stride=1,
                                       padding=1, groups=head_dim, bias=False)

        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.projection = nn.Sequential(nn.Conv2d(out_channels, projection_out_channels, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(out_channels),
                                        nn.ReLU(inplace=True))

        self.conv_scale = nn.Sequential(nn.Conv2d(in_channels=out_channels*3, out_channels=projection_out_channels, kernel_size=1, stride=1),
                                        nn.ReLU(inplace=True))

        self.conv_upsample = nn.Sequential(nn.Conv2d(in_channels=projection_out_channels, out_channels=channels_next_layer, kernel_size=1, stride=1),
                                        nn.BatchNorm2d(channels_next_layer),
                                        nn.ReLU(inplace=True))
        self.se = SEBlock(out_channels*3)

    def forward(self, x):

        out1 = self.conv1x1_1(x) #k
        out2 = self.conv1x1_2(x) #q
        out3 = self.conv1x1_3(x) #v

        output  = torch.cat((out1, out2, out3), 1)

        out = self.group_conv3x3(output)
        out = self.norm(out)
        out = self.act(out)
        out = self.projection(out)
        out_temp = self.se(output)
        out_temp = self.conv_scale(out_temp)

        out1 = out*out_temp
        out = self.conv_upsample(out1)

        return out
