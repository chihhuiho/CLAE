import torch.nn as nn
import torchvision
from .resnet_BN import *
from .resnet_BN_imagenet import *


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SimCLR_BN(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, args, bn_adv_flag=False, bn_adv_momentum = 0.01, data='non_imagenet'):
        super(SimCLR_BN, self).__init__()

        self.args = args
        self.bn_adv_flag = bn_adv_flag
        self.bn_adv_momentum = bn_adv_momentum    
        if data == 'imagenet':
            self.encoder = self.get_imagenet_resnet(args.resnet)
        else:
            self.encoder = self.get_resnet(args.resnet)

        self.n_features = self.encoder.fc.in_features  # get dimensions of fc layer
        self.encoder.fc = Identity()  # remove fully-connected layer after pooling layer
        
        self.projector = nn.Sequential(
                nn.Linear(self.n_features, self.n_features),
                nn.ReLU(),
                nn.Linear(self.n_features, args.projection_dim),
            )


    def get_resnet(self, name):
        resnets = {
            "resnet18": resnet18(pool_len=4, bn_adv_flag=self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum),
            "resnet34": resnet34(pool_len=4, bn_adv_flag=self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum),
            "resnet50": resnet50(pool_len=4, bn_adv_flag=self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum),
            "resnet101": resnet101(pool_len=4, bn_adv_flag=self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum),
            "resnet152": resnet152(pool_len=4, bn_adv_flag=self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum),
        }
        if name not in resnets.keys():
            raise KeyError(f"{name} is not a valid ResNet version")
        return resnets[name]
     
    def get_imagenet_resnet(self, name):
        resnets = {
            "resnet18": resnet18_imagenet(bn_adv_flag=self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum),
            "resnet34": resnet34_imagenet(bn_adv_flag=self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum),
            "resnet50": resnet50_imagenet(bn_adv_flag=self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum),
            "resnet101": resnet101_imagenet(bn_adv_flag=self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum),
            "resnet152": resnet152_imagenet(bn_adv_flag=self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum),
        }
        if name not in resnets.keys():
            raise KeyError(f"{name} is not a valid ResNet version")
        return resnets[name]



    def forward(self, x, adv=False):
        h = self.encoder(x, adv=adv)
        z = self.projector(h)

        if self.args.normalize:
            z = nn.functional.normalize(z, dim=1)
        return h, z
