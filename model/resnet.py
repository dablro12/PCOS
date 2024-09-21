import torch.nn as nn
from torchvision import models

class binary_model(nn.Module):
    def __init__(self):
        super(binary_model, self).__init__()
        self.base_model = models.resnet18(weights = models.ResNet18_Weights)
        # vgg16 마지막 분류기 부분을 바이너리 분류에 맞게 변경
        self.base_model.fc = nn.Linear(512, 1)
        
    def forward(self, x):
        return self.base_model(x).view(-1)

class multi_model(nn.Module):
    def __init__(self):
        super(multi_model, self).__init__()
        self.base_model = models.resnet18(weights = models.ResNet18_Weights)
        # vgg16 마지막 분류기 부분을 바이너리 분류에 맞게 변경
        self.base_model.fc = nn.Linear(512, 3)
        
    def forward(self, x):
        return self.base_model(x)
