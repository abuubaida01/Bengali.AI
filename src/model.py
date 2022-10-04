# we need pretrained model
# https://github.com/Cadene/pretrained-models.pytorch
import pretrainedmodels
import torch.nn as nn
from torch.nn import functional as F

class ResNet34(nn.Module):
    def __init__(self, pretrained):
        super(ResNet34, self).__init__()

        if pretrained is True:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained='imagenet')
        else: # validation time 
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained=None)
        #(last_linear): Linear(in_features=512, out_features=1000, bias=True); need to change
        self.l0 = nn.Linear(512, 168) # 168 Grapheme
        self.l1 = nn.Linear(512, 11) # 11 Vowels
        self.l2 = nn.Linear(512, 7) # 7 Consonants
    
    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1) # output size 1, reshape
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        return l0, l1, l2


