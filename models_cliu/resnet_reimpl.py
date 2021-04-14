import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleneckReimpl(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, out_planes, stride=1):
        super(BottleneckReimpl, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        #print(x.size())
        #print(self.conv1)
        #a = self.conv1(x)
        #print(a.size())
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetReimpl(nn.Module):
    '''
    input image size 32x 32

    '''
    def __init__(self, block, num_blocks):
        super(ResNetReimpl, self).__init__()
        self.in_planes = 64

        self.layer1 = nn.Conv2d(3,64,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer2 = self._make_layer(block, 64, 64, 256, num_blocks[0], stride=1)
        self.layer3 = self._make_layer(block, 256, 128, 512, num_blocks[1], stride=2)
        self.layer4 = self._make_layer(block, 512, 256, 1024, num_blocks[2], stride=2)
        self.layer5 = self._make_layer(block, 1024, 512, 2048, num_blocks[3], stride=2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=2048, out_features=10)

    def _make_layer(self, block, in_planes, planes, out_planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for ind,stride in enumerate(strides):
            if ind == 0:
                #print('g')
                # first res block will do feature map size reduction
                layers.append(block(in_planes, planes, out_planes, stride=strides[ind]))
            else:
                layers.append(block(out_planes, planes, out_planes, stride=strides[ind]))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.layer1(x))) # 32 * 32
        out = self.layer2(out) # 32
        out = self.layer3(out) # 16
        out = self.layer4(out) #  8
        out = self.layer5(out) # 4 x 4
        out = F.avg_pool2d(out, 4)
        out = self.flatten(out)
        #print(out.size())
        out = self.linear(out)
        return out

def ResNet50Reimpl():
    return ResNetReimpl(BottleneckReimpl, [3, 4, 6, 3])


def ResNet101Reimpl():
    return ResNetReimpl(BottleneckReimpl, [3, 4, 23, 3])