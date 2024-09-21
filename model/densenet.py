import torch
import torch.nn as nn 

class binary_model(nn.Module):
    """ ref : https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py """
    def __init__(self):
        super(binary_model, self).__init__()
        self.base_model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        self.base_model.classifier = nn.Linear(1024, 1)
    
    def forward(self, x):
        return self.base_model(x).view(-1)
    

class multi_model(nn.Module):
    """ ref : https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py """
    def __init__(self):
        super(multi_model, self).__init__()
        self.base_model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        self.base_model.classifier = nn.Linear(1024, 3)
    
    def forward(self, x):
        return self.base_model(x)
    
        
    