import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_plane, out_plane, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_plane,
                              out_channels=in_plane,
                              groups=in_plane,
                              kernel_size=3,
                              stride=stride,
                              padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=in_plane)
        self.conv2 = nn.Conv2d(in_channels=in_plane,
                               out_channels=out_plane,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        self.bn2 = nn.BatchNorm2d(num_features=out_plane)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu1(out)
        return out


class MobileNetReimpl(nn.Module):
    def __init__(self):
        super(MobileNetReimpl, self).__init__()
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)


        # store the out plan and stride
        self.cfg = [64,
               (128, 2),
               128,
               (256, 2),
               256,
               (512, 2),
               512, 512, 512, 512, 512,
               (1024, 2),
               1024]
        in_planes = 32
        self.layers = self._make_layers(in_planes)
        self.linear = nn.Linear(in_features=1024, out_features=10)
        self.pool = nn.AvgPool2d(kernel_size=2)

    def _make_layers(self, in_planes):
        layers = []

        for x in self.cfg:
            out_planes = (x if isinstance(x,int) else x[0])
            stride = (1 if isinstance(x, int) else x[1])
            layers.append(Block(
                in_plane=in_planes,
                out_plane=out_planes,
                stride=stride
            ))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out= self.bn(out)
        out = self.relu(out)
        out = self.layers(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out