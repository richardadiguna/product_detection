import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import MemoryEfficientSwish


class ResNet(nn.Module):
    def __init__(self, num_classes, unfreeze_block=True):
        super().__init__()
        self.num_classes = num_classes
        self.unfreeze_block = unfreeze_block
        
        resnet = models.resnet50(pretrained=True)
        for idx, parameter in enumerate(resnet.parameters()):
            parameter.requires_grad = False
            
        if self.unfreeze_block:
            for idx, block in enumerate(list(resnet.children())[6:8]):
                for param in block.parameters():
                    param.requires_grad = True       
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, 1028)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1028, self.num_classes)
        
    def forward(self, images):
        x = self.resnet(images)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
class EfficientNetwork(nn.Module):
    def __init__(self, num_classes, unfreeze_block):
        super().__init__()
        self.num_classess = num_classes
        self.unfreeze_block = unfreeze_block
        
        efficient_net = EfficientNet.from_pretrained(
            'efficientnet-b4', 
            num_classes=self.num_classess)
        
        for idx, block in enumerate(efficient_net.parameters()):
            param.requires_grad = False
            
        if self.unfreeze_block:
            for idx, block in enumerate(list(efficient_net.children())[3:]):
                for param in block.parameters():
                    param.requires_grad = True
        
        self.efficient_net = efficient_net
        
    def forward(self, images):
        x = self.efficient_net(images)
        return x
    
    
class AlexNet(nn.Module):
    def __init__(self, num_classes, unfreeze_block):
        super().__init__()
        self.num_classes = num_classes
        self.unfreeze_block = unfreeze_block
        
        self.alexnet = models.alexnet(pretrained=True)
        
    def forward(self, images):
        return