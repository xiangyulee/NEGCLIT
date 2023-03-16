from model.scalable_model import ScalableResNet
import torch
from torchvision import datasets, transforms
from fedlab.contrib.data.partitioned_cifar10 import PartitionedCIFAR10
from fedlab.contrib.data.partitioned_cifar100 import PartitionedCIFAR100

model_name={'resnet':ScalableResNet}
dataset_class_num={'cifar10':10,
                  'cifar100':100}
dataset_name={'cifar10':datasets.CIFAR10,
              'cifar100':datasets.CIFAR100}
online_dataset_name={'cifar10':PartitionedCIFAR10,
                    'cifar100':PartitionedCIFAR100}