import os
import torch
import torch.optim as optim
from fedlab.utils.logger import Logger
from fedlab.contrib.algorithm.fedavg import FedAvgServerHandler
from fedlab.core.server.manager import SynchronousServerManager,AsynchronousServerManager
from fedlab.core.network import DistNetwork
from experiment.SSH_client  import sock_server_data
from util.name_match import dataset_class_num,model_name
from util.method_match import model_weight_lighting,offline_train_method
def selfgrow_deploy(args):
    # split network to NE and NG
    net=torch.load(os.path.join(args.save_server,'model_best.pth.tar'))
    # model_name[args.model].growth=net['growth']
    model=model_name[args.model](net['nclass'])
    NE=model.NE
    NE.load_state_dict(net["NE_state_dict"])
    NE.cuda()
    NE=model_weight_lighting[args.prune](NE,args)
    NE=offline_train_method[args.train_method](args)
    sock_server_data(args)   # client的存储位置需要提前建立  代码中是本地 所以没有出现问题
    return NE

def autosplit_deploy(args):
    # split network to NE and NG
    net=torch.load(os.path.join(args.save_server,'model_best.pth.tar'))
    # model_name[args.model].growth=net['growth']
    model=model_name[args.model](net['nclass'],net['split_layer'])
    NE=model.NE
    NE.load_state_dict(net["NE_state_dict"])
    NE.cuda()
    sock_server_data(args)   # client的存储位置需要提前建立  代码中是本地 所以没有出现问题
    return NE

def fixedsplit_deploy(args):
    # split network to NE and NG
    net=torch.load(os.path.join(args.save_server,'model_best.pth.tar'))
    # model_name[args.model].growth=net['growth']
    model=model_name[args.model](net['nclass'])
    NE=model.NE
    NE.load_state_dict(net["NE_state_dict"])
    NE.cuda()
    sock_server_data(args)   # client的存储位置需要提前建立  代码中是本地 所以没有出现问题
    return NE
