"""
ResNet-18 PoseNet model for camera pose regression on Stanford 2D-3D-S dataset.
Predicts 3D position [x,y,z] and 4D normalized quaternion orientation [qw,qx,qy,qz].
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetPoseNet(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNetPoseNet, self).__init__()

        # Use torchvision ResNet-18 backbone
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
        except AttributeError:
            resnet = models.resnet18(pretrained=pretrained)

        # Remove the classification head (fc layer)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        # Feature size is 512 for ResNet-18
        self.position_head = nn.Linear(512, 3)
        self.orientation_head = nn.Linear(512, 4)

    def forward(self, x):
        features = self.resnet(x)
        features = features.view(features.size(0), -1)

        position = self.position_head(features)
        orientation = self.orientation_head(features)

        # Normalize quaternion to unit length: q = q / ||q||
        orientation = torch.nn.functional.normalize(orientation, p=2, dim=1)

        return position, orientation
