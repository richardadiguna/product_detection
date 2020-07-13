import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon, reduction):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, logits, labels):
        """
        Smooth label 
        """
        confidence = 1 - self.epsilon
        logprobs = F.log_softmax(logits, dim=-1)
        nll_loss = F.nll_loss(logprobs, labels, reduction='none')
        smooth_loss = -logprobs.sum(dim=-1)
        loss = confidence * nll_loss + self.epsilon * smooth_loss
        return loss.mean()
    
    
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, x, target):
        confidence = 1. - self.epsilon
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.epsilon * smooth_loss
        return loss.mean()
    

class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        return
    
    
class LearningRateWarmup(nn.Module):
    def __init__(self):
        super().__init__()
        return
        
    