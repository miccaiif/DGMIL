from torch import nn
import torch.nn.functional as F

class Linear_projection(nn.Module):
    def __init__(self):
        super(Linear_projection, self).__init__()
        self.projection_head = nn.Linear(512, 512)
        self.bn = nn.BatchNorm1d(512)
        self.class_layer = nn.Linear(512, 2)
    def forward(self, x):
        x = self.projection_head(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.class_layer(x)
        return x

class Linear_projection_MAE(nn.Module):
    def __init__(self):
        super(Linear_projection_MAE, self).__init__()
        self.projection_head = nn.Linear(768, 768)
        self.bn = nn.BatchNorm1d(768)
        self.class_layer = nn.Linear(768, 2)
    def forward(self, x):
        x = self.projection_head(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.class_layer(x)
        return x

class Linear_projection_256(nn.Module):
    def __init__(self):
        super(Linear_projection_256, self).__init__()
        self.projection_head = nn.Linear(256, 256)
        self.bn = nn.BatchNorm1d(256)
        self.class_layer = nn.Linear(256, 2)
    def forward(self, x):
        x = self.projection_head(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.class_layer(x)
        return x

class Linear_projection_768(nn.Module):
    def __init__(self):
        super(Linear_projection_768, self).__init__()
        self.projection_head = nn.Linear(768, 768)
        self.bn = nn.BatchNorm1d(768)
        self.class_layer = nn.Linear(768, 2)
    def forward(self, x):
        x = self.projection_head(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.class_layer(x)
        return x