from itertools import chain

import torch
import torch.nn.functional as F
import torch.nn as nn
from model.resnet import ResNet_E,BasicBlock,Bottleneck
import multiprocessing
import subprocess

# 创建早期退出分支
class EarlyExitBranch(nn.Module):
    def __init__(self, in_features, num_classes):
        super(EarlyExitBranch, self).__init__()
        self.branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.branch(x)

class WeightedResNet(nn.Module):
    def __init__(self, nclasses,split_layer=-1,input_shape=[1,3,32,32],cfg=None):
        super().__init__()
        self.ALL=ResNet_E(nclasses,cfg=cfg)
        self.cfg=cfg
        self.num_heads=count_layers(self.ALL)
        self.ws=nn.Parameter(
                torch.zeros(self.num_heads, dtype=torch.float, requires_grad=True))
        self.layer_penalties = torch.arange(0.0, self.num_heads) + 1.0
        self.NE=None
        self.NG=None
        self.ee=[]
        
        in_features=self.compute_in(input_shape)
        for i in range(self.num_heads):
            self.ee.append(EarlyExitBranch(in_features[i],nclasses))
        if split_layer!=-1:
            self.add_head_split(split_layer)
        
    # 计算每个早退分支输入特征数
    def compute_in(self,input_shape=[]):
        in_features=[1]*self.num_heads
        if input_shape:
            input_data = torch.randn(*input_shape)
            for i,sub in enumerate(self.ALL.children()):
                input_data=sub(input_data)
                in_features[i]=input_data.numel()
        return in_features
    def split(self,split_layer=-1):
        # 获取最大值及其下标
        if split_layer==-1:
            max_value, max_index = torch.max(self.ws, dim=0)  
        else:
            max_index=split_layer
        # 分割模型 
        NE_layer=list(self.ALL.children())[0:max_index+1]
        self.NE=nn.Sequential(*NE_layer)
        NG_layer=list(self.ALL.children())[max_index+1:]
        self.NG=nn.Sequential(*NG_layer)
        return max_index
    def add_head_split(self,split_layer=-1):
        index=self.split(split_layer)
        self.NE.add_module('pre_head',self.ee[index])
        def features(self,x):
            for name, layer in self.named_children():
                if name=='pre_head':
                    break
                x=layer(x)
            return x
        setattr(self.NE, 'features', features.__get__(self.NE))
        return index
    def forward(self,x):
        if not self.NE and not self.NG:
            x=self.ALL(x)
        else:
            
            x=self.NE(x)
            x=self.NG(x)
        return x
    def early_exit(self,x):
        y=[]
        for i,layer in enumerate(self.ALL.children()):
            x=layer(x)
            y.append(self.ee[i](x)*self.ws[i])

        return y


def count_layers(module, layer_types=None,recursive=False):
    if layer_types is None:
        layer_types = (nn.Module,)
    elif not isinstance(layer_types, tuple):
        layer_types = (layer_types,)

    layer_count = 0
    for child in module.children():
        if isinstance(child, layer_types):
            layer_count += 1
        if recursive:
            layer_count += count_layers(child, layer_types)
    return layer_count

class DCNet(nn.Module):

    def __init__(self, image_size, channels, num_layers, num_filters, kernel_size, classes, beta=0, gamma=False,
                 batchnorm=True, baseline=False):
        # TODO handle non-square images, different architectures, padding and stride if it becomes necessary
        super().__init__()
        assert image_size > 1
        assert channels >= 1
        assert classes > 1
        assert num_layers >= 1
        if baseline:
            assert not gamma
        self.image_size = image_size
        self.channels = channels
        self.num_heads = num_layers
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        # self.filters_inc = filters_inc
        self.classes = classes
        self.beta = beta
        self.gamma = gamma
        self.batchnorm = batchnorm
        self.baseline = baseline
        if not self.baseline:
            # weights for layers
            self.ws = nn.Parameter(
                torch.zeros(self.num_heads, device=get_device(), dtype=torch.float, requires_grad=True))
            self.layer_penalties = torch.arange(0.0, self.num_heads, device=get_device()) + 1.0
        # layer lists
        self.layers = nn.ModuleList()
        if self.batchnorm:
            self.bn_layers = nn.ModuleList()
        if not self.baseline:
            self.heads = nn.ModuleList()
        # assume, for simplicity, that we only use 'same' padding and stride 1
        padding = (self.kernel_size - 1) // 2
        # layers
        c_in = self.channels
        c_out = self.num_filters
        for layer in range(self.num_heads):
            self.layers.append(nn.Conv2d(c_in, c_out, kernel_size=self.kernel_size, stride=1, padding=padding))
            c_in, c_out = c_out, c_out
            # c_in, c_out = c_out, c_out + self.filters_inc
            if self.batchnorm:
                self.bn_layers.append(nn.BatchNorm2d(c_out))
            if not self.baseline:
                self.heads.append(nn.Linear(c_out, self.classes))
        if self.baseline:
            self.layers.append(nn.Linear(c_out, self.classes))
            # self.convs.append(nn.Linear(c_out * self.image_size ** 2, self.classes))

    def init_layer_importances(self, init='uniform'):
        # TODO deduplicate
        if not self.baseline:
            if isinstance(init, str):
                if init == 'uniform':
                    self.ws.data = torch.zeros(self.num_heads, device=get_device(), dtype=torch.float)
                elif init == 'first':
                    self.ws.data = torch.zeros(self.num_heads, device=get_device(), dtype=torch.float)
                    self.ws.data[0] = 10.0
                elif init == 'last':
                    self.ws.data = torch.zeros(self.num_heads, device=get_device(), dtype=torch.float)
                    self.ws.data[-1] = 10.0
                else:
                    raise ValueError('Incorrect init type')
            else:
                self.ws.data = init

    def parameters(self):
        to_chain = [self.layers]
        if self.batchnorm:
            to_chain.append(self.bn_layers)
        if not self.baseline:
            to_chain.append(self.heads)
        yield from chain.from_iterable(m.parameters() for m in chain.from_iterable(to_chain))

    def importance_parameters(self):
        return [self.ws, ]

    def non_head_parameters(self):
        if self.batchnorm:
            return (list(l.parameters()) + list(l_bn.parameters()) for l, l_bn in zip(self.layers, self.bn_layers))
        else:
            return (list(l.parameters()) for l in self.layers)

    def forward(self, x):
        if not self.baseline:
            layer_outputs = []
            for i, (conv_layer, c_head) in enumerate(zip(self.layers, self.heads)):
                x = torch.relu(conv_layer(x))
                if self.batchnorm:
                    x = self.bn_layers[i](x)
                x_transformed = nn.functional.max_pool2d(x, (x.size(2), x.size(3))).view(x.size(0), -1)
                layer_outputs.append(torch.log_softmax(c_head(x_transformed), dim=1))
            layer_outputs_tensor = torch.stack(layer_outputs, dim=2)
            y_pred = torch.matmul(layer_outputs_tensor, torch.softmax(self.ws, dim=0))
            if self.gamma:
                exped_sum = y_pred.exp().sum(dim=1, keepdim=True)
                assert torch.isnan(exped_sum).sum().item() == 0
                assert (exped_sum == float('inf')).sum().item() == 0
                assert (exped_sum == float('-inf')).sum().item() == 0
                # assert (exped_sum == 0.0).sum().item() == 0  # this throws
                log_exped_sum = (exped_sum + 1e-30).log()
                assert torch.isnan(log_exped_sum).sum().item() == 0
                assert (log_exped_sum == float('inf')).sum().item() == 0
                assert (log_exped_sum == float('-inf')).sum().item() == 0
                y_pred -= self.gamma * log_exped_sum
                assert torch.isnan(y_pred).sum().item() == 0
                assert (y_pred == float('inf')).sum().item() == 0
                assert (y_pred == float('-inf')).sum().item() == 0
            return y_pred, layer_outputs
        else:
            for i, layer in enumerate(self.layers[:-1]):
                x = torch.relu(layer(x))
                if self.batchnorm:
                    x = self.bn_layers[i](x)
            x_transformed = nn.functional.max_pool2d(x, (x.size(2), x.size(3))).view(x.size(0), -1)
            last_activations = self.layers[-1](x_transformed)
            return torch.log_softmax(last_activations, dim=1), []

    def calculate_loss(self, output, target, criterion):
        if self.baseline:
            return criterion(output, target), torch.tensor(0.0)
        else:
            pen = self.beta * torch.sum(torch.softmax(self.ws, dim=0) * self.layer_penalties)
            if isinstance(output, list):
                layer_outputs = output
                layer_outputs_tensor = torch.stack(layer_outputs, dim=2)
                output = torch.matmul(layer_outputs_tensor, torch.softmax(self.ws, dim=0))
            return criterion(output, target) + pen, pen

def get_gpu_memory():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,nounits,noheader'
        ])
    gpu_memory = [int(x) for x in result.decode().strip().split()]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def get_device():
    global device
    if device is None:
        print(f'{multiprocessing.cpu_count()} CPUs')

        print(f'{torch.cuda.device_count()} GPUs')
        if torch.cuda.is_available():
            device = 'cuda:0'
            # torch.set_default_tensor_type(torch.cuda.FloatTensor)
            for k, v in get_gpu_memory().items():
                print(f'Device {k} memory: {v} MiB')
            torch.backends.cudnn.benchmark = True
        else:
            # torch.set_default_tensor_type(torch.FloatTensor)
            device = 'cpu'
        print(f'Using: {device}')
    return device