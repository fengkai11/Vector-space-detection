from __future__ import absolute_import
from __future__ import division
from hydranets import HydraNets
from torchvision.models import resnet50
from transformer import Transformer
from torch import nn
import torch
class ConvBnRelu(nn.Module):
    def __init__(self,in_dim,out_dim,k,stride =1,with_bn =True):
        super(ConvBnRelu,self).__init__()
        pad = (k-1)//2
        self.conv = nn.Conv2d(in_dim,out_dim,(k,k),padding= (pad,pad),stride = (stride,stride))
        self.bn= nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace = True)
    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)
        return relu

class ResBlock(nn.Module):
    def __init__(self,in_dim):
        super(ResBlock,self).__init__()
        middle_dim = in_dim//4
        self.conv0 = ConvBnRelu(in_dim,middle_dim,1)
        self.conv1 = ConvBnRelu(middle_dim,middle_dim,3)
        self.conv2 = ConvBnRelu(middle_dim,in_dim,1)
    def forward(self,x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        return x+conv2


class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50,self).__init__()
        model = resnet50(pretrained= False)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1

        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        #infer size
        tmp = list(self.layer4)
        in_dim = tmp[0].conv3.weight.shape[0]
        self.resblock = ResBlock(in_dim)
        self.layer5 = nn.Sequential(self.resblock,self.resblock)

    def forward(self,x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        features = []
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        x = self.maxpool(x)
        x = self.layer5(x)
        features.append(x)
        return features

#modified from  https://github.com/aim-uofa/AdelaiDet/blob/master/adet/modeling/backbone/bifpn.py
class SingleBiFPN(nn.Module):
    def __init__(self,in_channels_list,out_channels,norm = ""):
        super(SingleBiFPN,self).__init__()
        if len(in_channels_list) == 5:
            self.nodes = [
                {'feat_level':3,'inputs_offsets':[3,4]},
                {'feat_level':2,'inputs_offsets':[2,5]},
                {'feat_level':1,'inputs_offsets':[1,6]},
                {'feat_level':0,'inputs_offsets':[0,7]},
                {'feat_level':1,'inputs_offsets':[1,7,8]},
                {'feat_level':2,'inputs_offsets':[2,6,9]},
                {'feat_level':3,'inputs_offsets':[3,5,10]},
                {'feat_level':4,'inputs_offsets':[4,11]},
            ]
        else:
            raise NotImplementedError()
        i = 0
    def forward(self,feats):
        num_levels = len(feats)
        num_output_connections = [0 for _ in feats]
        for fnode in self.nodes:
            feat_level = fnode['feat_level']
            inputs_offets = fnode['inputs_offsets']
            input_node = []
            _,_,target_h,target_w = feats[feat_level].size()
            for input_offset in inputs_offets:
                num_output_connections[input_offset]+=1
                input_node = feats[input_offset]










class BifNet(nn.Module):
    def __init__(self):
        super(Resnet50).__init__()
        i = 0
    def forwared(self,x):
        feature = x
        return feature
class HydraNetsResnet50(HydraNets):
    def __init__(self,extrator,neck,transformer,heads):
        super(HydraNetsResnet50).__init__(extrator,neck,transformer,heads)

if __name__ == "__main__":
    f = Resnet50()
    input = torch.randn(1,3,512,512)
    features = f.forward(input)
    for item in features:
        print(item.size())


