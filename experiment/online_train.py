import os
import torch
import torch.optim as optim
from fedlab.utils.logger import Logger
from fedlab.contrib.algorithm.selfgrow import SelfGrowServerHandler
from fedlab.contrib.algorithm.fixedsplit import FixedSplitServerHandler
from fedlab.contrib.algorithm.autosplit import AutoSplitServerHandler
from fedlab.core.server.manager import SynchronousServerManager,AsynchronousServerManager
from fedlab.core.network import DistNetwork

from util.name_match import dataset_class_num,model_name

def online_autosplit(args,NE=None):
    # server online training,including :
    # 1.communication with clients 
    # 2.train network graph
    # 3.inference non-early-exit instances depending on features from client 
    
    # model = deploy(params) # 考虑返回NE 
    
    # print(model) # R
    if NE==None:
        net=torch.load(os.path.join(os.getcwd(),f'result/server/model_best.pth.tar')) #path change
        model=model_name[args.model](nclasses=dataset_class_num[args.offline_dataset],split_layer=net['split_layer'],cfg=net['cfg'])
        model.load_state_dict(net['state_dict'])
        NE=model.NE
    if args.cuda:
        NE.cuda()
    args.optimizer = optim.SGD(NE.parameters(), 
                                lr=args.online_lr, momentum=args.online_momentum, weight_decay=args.online_weight_decay)
    LOGGER = Logger(log_name="server")


    handler = AutoSplitServerHandler(NE,
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
def online_fixedsplit(args,NE=None):
    # server online training,including :
    # 1.communication with clients 
    # 2.train network graph
    # 3.inference non-early-exit instances depending on features from client 
    
    # model = deploy(params) # 考虑返回NE 
    
    # print(model) # R
    if NE==None:
        net=torch.load(os.path.join(os.getcwd(),f'result/server/model_best.pth.tar')) #path change
        
        model=model_name[args.model](nclasses=dataset_class_num[args.offline_dataset],cfg=net['cfg'])
        model.load_state_dict(net['state_dict'])
        NE=model.NE
    if args.cuda:
        NE.cuda()
    args.optimizer = optim.SGD(NE.parameters(), 
                                lr=args.online_lr, momentum=args.online_momentum, weight_decay=args.online_weight_decay)
    LOGGER = Logger(log_name="server")


    handler = FixedSplitServerHandler(NE,
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

def online_selfgrow(args,NE=None):
    # server online training,including :
    # 1.communication with clients 
    # 2.train network graph
    # 3.inference non-early-exit instances depending on features from client 
    
    # model = deploy(params) # 考虑返回NE 
    
    # print(model) # R
    if NE==None:
        net=torch.load(os.path.join(os.getcwd(),f'result/server/model_best.pth.tar')) #path change
        model=model_name[args.model](nclasses=dataset_class_num[args.offline_dataset],growth=net['growth'],cfg=net['cfg']) 

        model.load_state_dict(net['state_dict'])
        NE=model.NE
    if args.cuda:
        NE.cuda()
    args.optimizer = optim.SGD(NE.parameters(), 
                                lr=args.online_lr, momentum=args.online_momentum, weight_decay=args.online_weight_decay)
    LOGGER = Logger(log_name="server")


    handler = SelfGrowServerHandler(NE,
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





