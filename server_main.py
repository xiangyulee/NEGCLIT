import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
import random
import numpy as np
import torch
from experiment.server import *
def main(args):
    print(args)
    # set up seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 
        #This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
    # offline_run(args)
    # deploy(args) 
    
    online_run(args)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Network Elements and Graph Cross Layer Inference and Training PyTorch")
    ########################General#########################
    parser.add_argument('--seed', dest='seed', default=1, type=int,
                        help='Random seed')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--save', default='result/', type=str, metavar='PATH',
                        help='path to save checkpoint (default: none)')
    parser.add_argument('--save_client', default='result/client/', type=str, metavar='PATH', #path change
                        help='path to save prune model (default: current directory)')
    parser.add_argument('--save_server', default='result/server/', type=str, metavar='PATH', #path change
                        help='path to save prune model (default: current directory)')
    parser.add_argument('--model', default='resnet', type=str, metavar='MODEL',
                        help='whole model:NE+NG(default:resnet)')           
    ########################Offline Training#########################
    parser.add_argument('--offline-dataset', type=str, default='cifar100',
                        help='training dataset (default: cifar10)')
    parser.add_argument('--s', type=float, default=0.0001,
                        help='scale sparse rate (default: 0)')
    parser.add_argument('--percent', type=float, default=0.3,
                        help='prune rate (default: 0.1)')
    parser.add_argument('--offline-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--offline-epoch', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--offline-lr', type=float, default=0.1, metavar='OFFLR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--offline-momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--offline-weight-decay', '--offwd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    ########################Online Training#########################
    parser.add_argument('--ip', type=str,default='127.0.0.1')
    parser.add_argument('--port', type=int,default=3001)
    parser.add_argument('--world_size', type=int,default=2)
    parser.add_argument('--ethernet', type=str, default=None)

    parser.add_argument('--round', type=int, default=100)
    parser.add_argument('--sample', type=float, default=1)
    parser.add_argument('--update', default='random', choices=['random', 'GSS', 'ASER'],
                        help='Update method  (default: %(default)s)')
    parser.add_argument('--retrieve', default='random', choices=['MIR', 'random', 'ASER', 'match', 'mem_match'],
                        help='Retrieve method  (default: %(default)s)')
    parser.add_argument('--online-lr', type=float, default=0.001, metavar='ONLR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--online-epoch', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--online-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--online-momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--online-weight-decay', '--onwd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')

    
    
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    main(args)
    
