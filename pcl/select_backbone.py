from .s3d import S3D
from torchvision import models
from torch import nn


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def select_backbone(net2D, net3D, first_channel=3):
    # create 2D backbone
    backbone2D = models.__dict__[net2D]()
    feat_size_2D = backbone2D.fc.weight.shape[1]
    backbone2D.fc = Identity()

    # create 3D backbone
    backbone3D = S3D()
    backbone3D = nn.Sequential(
                            backbone3D,
                            nn.AdaptiveAvgPool3d((1, 1, 1)))

    feature_size = dict()
    feature_size['3D'] = 1024
    feature_size['2D'] = feat_size_2D

    return backbone2D, backbone3D, feature_size
