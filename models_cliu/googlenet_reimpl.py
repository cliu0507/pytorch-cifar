import torch
import torch.nn as nn

class InceptionV1Reimpl(nn.Module):
    '''
    Reimplementation of inception module with dimension reductions
    https://arxiv.org/pdf/1409.4842.pdf
    '''
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes ):
        super(InceptionV1Reimpl, self).__init__()
        # 1 x 1 conv branch
        self.b1 = nn.Sequential(nn.Conv2d(in_channels=in_planes, out_channels= n1x1, kernel_size=1, bias=False),
                                nn.BatchNorm2d(n1x1),
                                nn.ReLU(True))

        # 1 x 1 conv  => 3 x 3 conv branch
        self.b2 = nn.Sequential(nn.Conv2d(in_channels=in_planes, out_channels= n3x3red, kernel_size=1),
                                nn.BatchNorm2d(n3x3red),
                                nn.ReLU(True),
                                nn.Conv2d(in_channels=n3x3red, out_channels=n3x3, kernel_size=3, padding=1),
                                nn.BatchNorm2d(n3x3),
                                nn.ReLU(True))

        # 1x1 conv -> 5x5 conv branch
        # basically use two 3x3 to replace 5x5
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self,x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)

class GoogleNetReimpl(nn.Module):
    '''
    assume input image is 32 x 32,
    so first couple layers don't use exact the same layers in the paper
    '''
    def __init__(self):
        super(GoogleNetReimpl, self).__init__()

        # res (batch, 192, 32, 32)
        self.conv1 = ConvBlock(in_channels=3, out_channels=192, kernel_size=3, padding=1, stride=1)

        # res (batch, 256, 32, 32)
        self.a3 = InceptionV1Reimpl(in_planes=192, n1x1=64, n3x3red=96, n3x3=128, n5x5red=16, n5x5=32, pool_planes=32)

        # (batch, 480, 32, 32)
        self.b3 = InceptionV1Reimpl(256, 128, 128, 192, 32, 96, 64)

        # (batch, 480, 16, 16)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)


        self.a4 = InceptionV1Reimpl(480, 192, 96, 208, 16, 48, 64)
        self.b4 = InceptionV1Reimpl(512, 160, 112, 224, 24, 64, 64)
        self.c4 = InceptionV1Reimpl(512, 128, 128, 256, 24, 64, 64)
        self.d4 = InceptionV1Reimpl(512, 112, 144, 288, 32, 64, 64)
        self.e4 = InceptionV1Reimpl(528, 256, 160, 320, 32, 128, 128)

        # # (batch, 832, 8, 8)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.a5 = InceptionV1Reimpl(832, 256, 160, 320, 32, 128, 128)
        self.b5 = InceptionV1Reimpl(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool1(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool2(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=stride,
                              bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.LeakyReLU(True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out