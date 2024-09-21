import torch.nn as nn
import timm 
class binary_model(nn.Module):
    def __init__(self):
        super(binary_model, self).__init__()
        # self.base_model = models.swin_v2_b(weights = models.Swin_V2_B_Weights)
        # vgg16 마지막 분류기 부분을 바이너리 분류에 맞게 변경
        self.base_model = timm.create_model('swin_small_patch4_window7_224.ms_in22k', pretrained=True,  num_classes=1)

    def forward(self, x):
        return self.base_model(x).view(-1)
    
class multi_model(nn.Module):
    def __init__(self):
        super(multi_model, self).__init__()
        # self.base_model = models.swin_v2_b(weights = models.Swin_V2_B_Weights)
        # vgg16 마지막 분류기 부분을 바이너리 분류에 맞게 변경
        self.base_model = timm.create_model('swin_small_patch4_window7_224.ms_in22k', pretrained=True,  num_classes=3)

    def forward(self, x):
        return self.base_model(x)