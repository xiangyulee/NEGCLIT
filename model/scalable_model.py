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
    
class ScalableNG(ScalableNetwork):
    
    def __init__(self,in_planes, nclasses ,growth="",cfg=None):
        super(ScalableNG, self).__init__()
        self.growth=growth
        self.nclasses=nclasses
        self.NG_pred=nn.Linear(in_planes,nclasses)
        self.NG_block=self.search_space[0](in_planes,in_planes)
        for i in range(len(self.growth)):
            self.NG_block.add_module('layer'+str(i),
            self.search_space[0](in_planes,in_planes))
    def grow(self,in_planes,name='layer'):   
        self.NG_block.add_module(name+str(self.growth),
        self.search_space[0](in_planes,in_planes))
        self.growth+='0'
    def forward(self, x):
        x = self.NG_block(x)
        x = x.view(x.size(0),-1)
        logits=self.NG_pred(x)
        return logits
    
class ScalableResNet(ScalableNetwork):
    def __init__(self, nclasses ,cfg=None):
        super(ScalableResNet, self).__init__()
        self.nclasses=nclasses
        self.NE=ResNet_E(nclasses,cfg=cfg)
        self.NG=ScalableNG(self.NE.features_planes,nclasses,growth="",cfg=cfg)

        self.cfg=cfg
    def grow(self,name='layer'): 
        in_planes=self.NE.features_planes     
        self.NG.grow(in_planes,name=name)
    def forward(self, x):
        x = self.NE.features(x)
        x = self.NG(x)
        return x
