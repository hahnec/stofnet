import torch
import torch.nn as nn


class ZonziniNetLarge(nn.Module):
    def __init__(self):
        super(ZonziniNetLarge, self).__init__()

        # feature extraction layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv1d(1, 50, kernel_size=10, stride=2))
        self.conv_layers.append(nn.Conv1d(50, 100, kernel_size=10, stride=2))
        self.conv_layers.append(nn.Conv1d(100, 150, kernel_size=10, stride=2))
        self.conv_layers.append(nn.Conv1d(150, 200, kernel_size=10, stride=2))
        self.conv_layers.append(nn.Conv1d(200, 250, kernel_size=10, stride=2))
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)

        # classification layers
        self.fc1 = nn.Linear(250, 1024)
        self.fc2 = nn.Linear(1024, 1)
        
    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = self.relu(conv_layer(x))
            x = self.maxpool(x)
        
        x = self.global_avgpool(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


class ZonziniNetSmall(nn.Module):
    def __init__(self):
        super(ZonziniNetSmall, self).__init__()

        # feature extraction layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv1d(1, 16, kernel_size=10, stride=2))
        self.conv_layers.append(nn.Conv1d(16, 32, kernel_size=10, stride=2))
        self.conv_layers.append(nn.Conv1d(32, 64, kernel_size=10, stride=2))
        self.conv_layers.append(nn.Conv1d(64, 64, kernel_size=10, stride=2))
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)

        # classification layers
        self.fc1 = nn.Linear(64, 1024)
        self.fc2 = nn.Linear(1024, 1)
        
    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = self.relu(conv_layer(x))
            x = self.maxpool(x)
        
        x = self.global_avgpool(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x