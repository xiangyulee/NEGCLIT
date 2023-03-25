import os
import torch
import torch.optim as optim
from fedlab.utils.logger import Logger
from fedlab.contrib.algorithm.fedavg import FedAvgServerHandler
from fedlab.core.server.manager import SynchronousServerManager,AsynchronousServerManager
from fedlab.core.network import DistNetwork
from experiment.SSH_client  import sock_server_data
from util.name_match import dataset_class_num,model_name
from util.method_match import model_weight_lighting,train_method

def offline_run(args,NE=None):
    train_method[args.train_method](args)
    
def deploy(args):
    # split network to NE and NG
    net=torch.load(os.path.join(args.save_server,'model_best.pth.tar'))
    # model_name[args.model].growth=net['growth']
    model=model_name[args.model](net['nclass'])
    NE=model.NE
    NE.load_state_dict(net["NE_state_dict"])
    NE.cuda()
    NE=model_weight_lighting[args.prune](NE,args)
    NE=offline_run(args,NE)
    sock_server_data(args)   # client的存储位置需要提前建立  代码中是本地 所以没有出现问题
    return NE


def online_run(args,NE=None):
    # server online training,including :
    # 1.communication with clients 
    # 2.train network graph
    # 3.inference non-early-exit instances depending on features from client 
    
    # model = deploy(params) # 考虑返回NE 
    
    # print(model) # R
    if NE==None:
        net=torch.load(os.path.join(os.getcwd(),f'result/server/model_best.pth.tar')) #path change
        if 'growth' in net.keys():
            model=model_name[args.model](nclasses=dataset_class_num[args.offline_dataset],growth=net['growth'],cfg=net['cfg']) 
        else:
            model=model_name[args.model](nclasses=dataset_class_num[args.offline_dataset],cfg=net['cfg'])
        model.load_state_dict(net['state_dict'])
        NE=model.NE
    if args.cuda:
        NE.cuda()
    args.optimizer = optim.SGD(NE.parameters(), 
                                lr=args.online_lr, momentum=args.online_momentum, weight_decay=args.online_weight_decay)
    LOGGER = Logger(log_name="server")


    handler = FedAvgServerHandler(NE,
                                        global_round=args.round,
                                        logger=LOGGER,
                                        sample_ratio=args.sample,
                                        args=args)


    network = DistNetwork(address=(args.ip, args.port),
                        world_size=args.world_size,
                        rank=0,
                        ethernet=args.ethernet)

    manager_ = SynchronousServerManager(handler=handler,
                                        network=network,
                                        mode = 'GLOBAL',
                                        logger=LOGGER)
    manager_.run(args)






