import torch.nn as nn
from model.resnet import ResNet_E,ResNet_G
class ScalableNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self,x):
        raise NotImplementedError()
    def grow(self):
        raise NotImplementedError()

class ScalableResNet(ScalableNetwork):
    growth=0
    def __init__(self, nclasses ,cfg=None):
        super(ScalableResNet, self).__init__()
        self.nclasses=nclasses
        self.NE=ResNet_E(nclasses,cfg=cfg)
        self.NG=ResNet_G(nclasses,self.NE.features_planes)
        for i in range(ScalableResNet.growth):
            self.NG.add_module('layer'+str(i),
            nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.nclasses, self.nclasses)
        ))
        self.cfg=cfg
    def grow(self,name='layer'):
            
        self.NG.add_module(name+str(ScalableResNet.growth),
        nn.Sequential(
        nn.ReLU(),
        nn.Linear(self.nclasses, self.nclasses)
    ))
        ScalableResNet.growth+=1
    def forward(self, x):
        out = self.NE.features(x)
        logits = self.NG(out)
        return logits