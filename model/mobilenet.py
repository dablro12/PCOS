
import torch.nn as nn
from torchvision import models

class binary_model(nn.Module):
    def __init__(self):
        super(binary_model, self).__init__()
        self.base_model = models.mobilenet_v2(weights = models.MobileNet_V2_Weights)
        # vgg16 마지막 분류기 부분을 바이너리 분류에 맞게 변경
        self.base_model.classifier[-1] = nn.Linear(1280, 1)
        
    def forward(self, x):
        return self.base_model(x).view(-1)

class multi_model(nn.Module):
    def __init__(self):
        super(multi_model, self).__init__()
        self.base_model = models.mobilenet_v2(weights = models.MobileNet_V2_Weights)
        # vgg16 마지막 분류기 부분을 바이너리 분류에 맞게 변경
        self.base_model.classifier[-1] = nn.Linear(1280, 3)
        
    def forward(self, x):
        return self.base_model(x)