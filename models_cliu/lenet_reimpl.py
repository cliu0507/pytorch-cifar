import torch
import torch.nn as nn

class LeNetReimpl(nn.Module):
    '''
    LeNet simple implementation for 3 channel image
    reference: https://www.cnblogs.com/silence-cho/p/11620863.html
    '''
    def __init__(self):
        super(LeNetReimpl, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6,padding=0, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,padding=0,kernel_size=5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, padding=0)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, padding=0, kernel_size=5)
        self.relu3 = nn.ReLU()
        self.flatten1 = nn.Flatten(start_dim=1, end_dim=-1)

        self.fc4 = nn.Linear(in_features=120, out_features=84, bias=True)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(in_features=84, out_features=10, bias=True)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu2(self.conv2(x))
        x = self.maxpool2(x)
        x = self.relu3(self.conv3(x))
        x = self.flatten1(x)
        x = self.relu4(self.fc4(x))
        x = self.fc5(x)
        return x