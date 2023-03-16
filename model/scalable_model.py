import torch.nn as nn
from model.resnet import ResNet_E,BasicBlock,Bottleneck
class ScalableNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.search_space=[BasicBlock,Bottleneck]
    def forward(self,x):
        raise NotImplementedError()
    def grow(self):
        raise NotImplementedError()

class ScalableResNet(ScalableNetwork):
    growth=""
    def __init__(self, nclasses ,cfg=None):
        super(ScalableResNet, self).__init__()
        self.nclasses=nclasses
        self.NE=ResNet_E(nclasses,cfg=cfg)
        self.NG_pred=nn.Linear(self.NE.features_planes,nclasses)
        self.NG=self.search_space[0](self.NE.features_planes,self.NE.features_planes)
        for i in range(len(ScalableResNet.growth)):
            self.NG.add_module('layer'+str(i),
            self.search_space[0](self.NE.features_planes,self.NE.features_planes))
        self.cfg=cfg
    def grow(self,name='layer'):
            
        self.NG.add_module(name+str(ScalableResNet.growth),
        self.search_space[0](self.NE.features_planes,self.NE.features_planes))
        ScalableResNet.growth+='0'
    def forward(self, x):
        x = self.NE.features(x)
        x = self.NG(x)
        x = x.view(x.size(0),-1)
        logits=self.NG_pred(x)
        return logits