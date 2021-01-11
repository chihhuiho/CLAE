import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable


    
class MySequential(nn.Sequential):
    def forward(self, x, adv):
        for module in self._modules.values():
            x = module(x, adv=adv)
        return x
    
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=False, expansion=0, bn_adv_flag=False, bn_adv_momentum=0.01):
        super(BasicBlock, self).__init__()
        self.bn_adv_momentum = bn_adv_momentum
        self.bn_adv_flag = bn_adv_flag
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        if self.bn_adv_flag:
            self.bn1_adv = nn.BatchNorm2d(planes, momentum = self.bn_adv_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        
        self.bn2 = nn.BatchNorm2d(planes)
        if self.bn_adv_flag:
            self.bn2_adv = nn.BatchNorm2d(planes, momentum = self.bn_adv_momentum)
            
        self.downsample = downsample
        if self.downsample: 
            self.ds_conv1 = nn.Conv2d(inplanes, planes * expansion, kernel_size=1, stride=stride, bias=False)
            self.ds_bn1 = nn.BatchNorm2d(planes*expansion)
            self.ds_bn1_adv = nn.BatchNorm2d(planes*expansion)
        self.stride = stride

    def forward(self, x, adv = False):
        residual = x
        if adv and self.bn_adv_flag:
            out = self.conv1(x)
            out = self.bn1_adv(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2_adv(out)
            if self.downsample:
                
                residual = self.ds_bn1_adv(self.ds_conv1(x))
            out += residual
            out = self.relu(out)
        else:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample:
                residual = self.ds_bn1(self.ds_conv1(x))
            out += residual
            out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=False, expansion=0, bn_adv_flag=False, bn_adv_momentum=0.01):
        super(Bottleneck, self).__init__()
        self.bn_adv_flag = bn_adv_flag
        self.bn_adv_momentum = bn_adv_momentum
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if self.bn_adv_flag:
            self.bn1_adv = nn.BatchNorm2d(planes, momentum = self.bn_adv_momentum)
            
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if self.bn_adv_flag:
            self.bn2_adv = nn.BatchNorm2d(planes, momentum = self.bn_adv_momentum)
            
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        if self.bn_adv_flag:
            self.bn3_adv = nn.BatchNorm2d(self.expansion*planes, momentum = self.bn_adv_momentum)
            
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        if self.downsample:
            self.ds_conv1 = nn.Conv2d(inplanes, planes * expansion, kernel_size=1, stride=stride, bias=False)
            self.ds_bn1 = nn.BatchNorm2d(planes * expansion)
            self.ds_bn1_adv = nn.BatchNorm2d(planes * expansion)
        self.stride = stride

    def forward(self, x, adv = False):
        residual = x
        if adv and self.bn_adv_flag:     
            out = self.conv1(x)
            out = self.bn1_adv(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2_adv(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3_adv(out)

            if self.downsample:
                
                residual = self.ds_bn1_adv(self.ds_conv1(x))

            out += residual
            out = self.relu(out)
        else:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample:
                
                residual = self.ds_bn1(self.ds_conv1(x))

            out += residual
            out = self.relu(out)
        return out


class ResNetAdvProp_imgnet(nn.Module):

    def __init__(self, block, layers, low_dim=128, is_feature=None, bn_adv_flag=False, bn_adv_momentum=0.01):
        super(ResNetAdvProp_imgnet, self).__init__()
        self.inplanes = 64
        self.bn_adv_flag = bn_adv_flag
        self.bn_adv_momentum = bn_adv_momentum
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        if bn_adv_flag:
            self.bn1_adv = nn.BatchNorm2d(64, momentum = self.bn_adv_momentum)
            
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], bn_adv_flag = self.bn_adv_flag, bn_adv_momentum=bn_adv_momentum)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, bn_adv_flag = self.bn_adv_flag, bn_adv_momentum=bn_adv_momentum)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, bn_adv_flag = self.bn_adv_flag, bn_adv_momentum=bn_adv_momentum)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, bn_adv_flag = self.bn_adv_flag, bn_adv_momentum=bn_adv_momentum)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, low_dim)
        self.dropout = nn.Dropout(p=0.5)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1, bn_adv_flag=False, bn_adv_momentum=0.01):
        downsample = False
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = True
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, expansion=block.expansion , bn_adv_flag=bn_adv_flag, bn_adv_momentum = bn_adv_momentum))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_adv_flag=bn_adv_flag, bn_adv_momentum = bn_adv_momentum))

        return MySequential(*layers)

    def forward(self, x, adv = False):
        
        x = self.conv1(x)
        if adv and self.bn_adv_flag:
            out = self.bn1_adv(x)
        else:
            out = self.bn1(x)
            
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, adv=adv)
        x = self.layer2(x, adv=adv)
        x = self.layer3(x, adv=adv)
        x = self.layer4(x, adv=adv)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet18_imagenet(low_dim=128, bn_adv_flag=False,bn_adv_momentum=0.01):
    return ResNetAdvProp_imgnet(BasicBlock, [2,2,2,2], low_dim, bn_adv_flag=bn_adv_flag, bn_adv_momentum=bn_adv_momentum)

def resnet34_imagenet(low_dim=128, bn_adv_flag=False, bn_adv_momentum=0.01):
    return ResNetAdvProp_imgnet(BasicBlock, [3,4,6,3], low_dim, bn_adv_flag=bn_adv_flag, bn_adv_momentum=bn_adv_momentum)

def resnet50_imagenet(low_dim=128, bn_adv_flag=False, bn_adv_momentum=0.01):
    return ResNetAdvProp_imgnet(Bottleneck, [3,4,6,3], low_dim, bn_adv_flag=bn_adv_flag, bn_adv_momentum=bn_adv_momentum)

def resnet101_imagenet( low_dim=128, bn_adv_flag=False, bn_adv_momentum=0.01):
    return ResNetAdvProp_imgnet(Bottleneck, [3,4,23,3], low_dim, bn_adv_flag=bn_adv_flag, bn_adv_momentum=bn_adv_momentum)

def resnet152_imagenet(low_dim=128, bn_adv_flag=False, bn_adv_momentum=0.01):
    return ResNetAdvProp_imgnet(Bottleneck, [3,8,36,3], low_dim, bn_adv_flag=bn_adv_flag, bn_adv_momentum=bn_adv_momentum)


def test():
    net = resnet50()
    # y = net(Variable(torch.randn(1,3,32,32)))
    # pdb.set_trace()
    y = net(Variable(torch.randn(1,3,224,224)), adv=True)
    # pdb.set_trace()
    print(y.size())
#test()

