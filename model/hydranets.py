from __future__ import absolute_import
from __future__ import division
from torch import nn
class HydraNets(nn.Module):
    def __init__(self,backbone,neck,transformer,heads):
        super(HydraNets).__init__()
        self.backbone = backbone
        self.neck = neck
        self.transformer = transformer
        self.heads = heads
    def forward(self,x):
        feature_list = self.backbone(x)
        feature = self.neck(feature_list)
        grid_feature = self.transformer(feature)
        results = self.heads(grid_feature)




