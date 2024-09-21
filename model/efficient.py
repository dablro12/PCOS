import torch.nn as nn
from torchvision import models

class pretrained_efficient_binary(nn.Module):
    def __init__(self):
        super(pretrained_efficient_binary, self).__init__()
        self.base_model = models.efficientnet_v2_s(weights = models.EfficientNet_V2_S_Weights)
        # vgg16 마지막 분류기 부분을 바이너리 분류에 맞게 변경
        self.base_model.classifier[-1] = nn.Linear(1280, 1)
        
    def forward(self, x):
        return self.base_model(x)