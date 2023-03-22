import argparse
import os

import sys
from util.name_match import model_name
import torch
from util.prune import prune_channel
from torchvision import transforms
from fedlab.core.client.manager import PassiveClientManager
from fedlab.core.network import DistNetwork
from fedlab.utils.logger import Logger
from fedlab.contrib.data.partitioned_cifar import PartitionCIFAR
from fedlab.contrib.data.partitioned_cifar10 import PartitionedCIFAR10
from fedlab.contrib.algorithm.fedavg import FedAvgClientTrainer

def build():
    # start this node and keep training data with labels or inferencing data without labels
    # communicate with server if necessary
    pass

def train():
    pass

def inference():
    pass

def deploy(params):
    # split network to NE and NG
    net=torch.load('result/model_best.pth.tar')
    model=model_name[params.model](net['nclass'])
    model.load_state_dict(net["state_dict"])
    NE=model.NE
    # NE.cuda()
    # NE=prune_channel(NE,0.5,params)
    return NE

