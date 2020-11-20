'''
Implementation of Baseline LeNet-5 on RGB Input and modified input size and modified output vector size (50 instead of 10)
Assume input height and width of 56x56 
'''
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(11*11*16, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 50)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(2,stride=2)

    def forward(self, x):
        out = self.relu(self.conv1(x)) # output HXW is 52
        out = self.avgpool(out) # output HXW is 26
        out = self.relu(self.conv2(out)) # output HXW is 22
        out = self.avgpool(out) # output HXW is 11
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        raw_predictions = self.fc3(out)
        return raw_predictions
