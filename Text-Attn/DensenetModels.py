import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import torchvision

class DenseNet121(nn.Module):

    def __init__(self, classCount, isTrained):
	
        super(DenseNet121, self).__init__()
		
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)

        kernelCount = self.densenet121.classifier.in_features
		
        self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet121(x)
        return x


class DenseNet121_Text(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size, text_rep_size=None, attn_size=None):
        super(DenseNet121_Text, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

        if text_rep_size is not None:
            assert attn_size is not None
            self.W_img = nn.Linear(num_ftrs, attn_size)
            self.W_text = nn.Linear(text_rep_size, attn_size)
            self.W_attn = nn.Linear(attn_size, 1)

    def forward(self, x, text_rep=None):
        features = self.densenet121.features(x)
        out = F.relu(features, inplace=True)
        if text_rep is None:
            out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        else:
            out = out.view(out.size(0), out.size(1), -1)
            out_t = torch.transpose(out, 1, 2)
            attn = F.softmax(self.W_attn(torch.tanh(self.W_img(out_t) +
                                                self.W_text(text_rep.unsqueeze(1).repeat(1, out.size(-1), 1)))), dim=1)
            out = torch.bmm(out, attn).squeeze()

        out = self.densenet121.classifier(out)
        return out


class TFIDFRep(nn.Module):
    """
    this class converts the tf-idf features into a representation hidden state vector
    """
    def __init__(self, feature_size, hidden_size, text_rep_size):
        super(TFIDFRep, self).__init__()
        self.W1 = nn.Linear(feature_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, text_rep_size)

    def forward(self, input):
        return self.W2(gelu(self.W1(input)))


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class DenseNet169(nn.Module):
    
    def __init__(self, classCount, isTrained):
        
        super(DenseNet169, self).__init__()
        
        self.densenet169 = torchvision.models.densenet169(pretrained=isTrained)
        
        kernelCount = self.densenet169.classifier.in_features
        
        self.densenet169.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        
    def forward (self, x):
        x = self.densenet169(x)
        return x
    
class DenseNet201(nn.Module):
    
    def __init__ (self, classCount, isTrained):
        
        super(DenseNet201, self).__init__()
        
        self.densenet201 = torchvision.models.densenet201(pretrained=isTrained)
        
        kernelCount = self.densenet201.classifier.in_features
        
        self.densenet201.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        
    def forward (self, x):
        x = self.densenet201(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, classCount, isTrained):
    
        super(ResNet50, self).__init__()
        
        self.resnet50 = torchvision.models.resnet50(pretrained=isTrained)

        kernelCount = self.resnet50.classifier.in_features
        
        self.resnet50.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.resnet50(x)
        return x