import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG_reimpl(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_reimpl, self).__init__()
        if vgg_name not in cfg:
            raise Exception(str(vgg_name) + ' is not in implemented VGG network list')
        self.features = self._make_layers(vgg_name)
        '''
        # since it is cifar 10 so don't need below
        self.fc1 = nn.Linear(in_features=512, out_features=4096)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=4096,out_features=4096)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(in_features=4096,out_features=1000)
        '''
        #self.fc1= nn.Linear(512, 4096)
        self.fc = nn.Linear(512, 10)


    def _make_layers(self, vgg_name):
        layers = []
        in_channels = 3
        for x in cfg[vgg_name]:
            if isinstance(x,int):
                layers.append(nn.Conv2d(in_channels=in_channels, out_channels=x, padding=1,kernel_size=3))
                layers.append(nn.BatchNorm2d(x))
                #layers.append(nn.ReLU())
                #layers.append(nn.Sigmoid())
                layers.append(nn.LeakyReLU())
                in_channels = x
            elif isinstance(x, str):
                if x == 'M':
                    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    raise Exception('Wrong layer type')
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        # another way of flatten tensor
        out = out.view(out.size(0), -1)
        #out = self.fc1(out)
        out = self.fc(out)
        return out


