import torch
import torch.nn as nn

class emnist_net(nn.Module):

    def __init__(self):
        super(emnist_net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        out = self.net(x)
        return out

class cifar10_net(nn.Module):

    def __int__(self):
        super(cifar10_net, self).__int__()
        self.Conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.Conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.Conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.Conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.Conv6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.Linear = nn.Linear(1024, 10)

    def forward(self, x):
        out = self.Conv1(x)
        out = self.Conv2(out)
        out = self.Conv3(out)
        out = self.Conv4(out)
        out = self.Conv5(out)
        out = self.Conv6(out)
        out = self.Linear(out)
        return out