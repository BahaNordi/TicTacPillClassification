# This is a simple network only with 8 conv layers

import torch.nn as nn
import torch.nn.functional as F


class VggBase(nn.Module):
    def __init__(self):
        super(VggBase, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3)
        self.bn6 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(576, 128)
        # self.dropout = nn.Dropout(p=0.5)
        self.bn7 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.bn2(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.bn3(self.conv3(x)), 2))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(F.max_pool2d(self.bn6(self.conv6(x)), 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.bn7(self.fc1(x)))
        # x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, -1)
