'''
Implementation of Baseline LeNet-5 on RGB Input and modified input size and modified output vector size (50 instead of 10)
Assume input height and width of 56x56
'''
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self,input_size):
        super(LeNet, self).__init__()
        self.input_size = input_size
        self.fc_feature_sz = self.calculate_feature_size()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(int(self.fc_feature_sz)*int(self.fc_feature_sz)*16, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 50)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(2,stride=2)

    def calculate_feature_size(self):
        f_conv_1 = self.input_size - 3 + 1
        f_conv_1_pooled = ((f_conv_1 - 2)/2) + 1
        f_conv_2 = f_conv_1_pooled - 6 + 1
        f_conv_2_pooled = ((f_conv_2 - 2)/2) + 1
        return f_conv_2_pooled

    def forward(self, x):
        out = self.relu(self.conv1(x)) # output HXW is 94
        out = self.avgpool(out) # output HXW is 47
        out = self.relu(self.conv2(out)) # output HXW is 42
        out = self.avgpool(out) # output HXW is 21
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        raw_predictions = self.fc3(out)
        return raw_predictions

a = LeNet(96)
