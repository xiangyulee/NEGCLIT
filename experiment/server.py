import os
import numpy as np
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from util.name_match import model_name
from util.prune import prune_channel
from fedlab.utils.logger import Logger
from fedlab.contrib.algorithm.fedavg import FedAvgServerHandler
from fedlab.core.server.manager import SynchronousServerManager,AsynchronousServerManager
from fedlab.core.network import DistNetwork
from experiment.SSH_client  import sock_server_data
from util.name_match import dataset_class_num,dataset_name
import time
import heapq

def save_checkpoint(state, is_best, filepath): 
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

def updateBN(model,s):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if m.weight.grad!=None:
                m.weight.grad.data.add_(s*torch.sign(m.weight.data))  # L1

def offline_train(epoch,model,args,optimizer,train_loader):
    start=time.time()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        ee_output=model.NE(data)
        loss = F.cross_entropy(output, target)+F.cross_entropy(ee_output, target)*0.2
        loss.backward()
        if args.s!=0:
            updateBN(model,args.s)
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {}/{}   [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch,args.offline_epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    end=time.time()
    print(f'train time:{end-start}')

def test(model,args,test_loader):
    model.eval()
    
    test_loss = 0
    correct = 0
    ee_correct=0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        ee_output=model.NE(data)
        # test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        ee_pred= ee_output.data.max(1, keepdim=True)[1] 
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        ee_correct+= ee_pred.eq(target.data.view_as(ee_pred)).cpu().sum()


    test_loss /= len(test_loader.dataset)
    print('\nTest set:Final Average Accuracy: {}/{} ({:.1f}%)\n'.format(
         correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('\nTest set:Early Exit Average Accuracy: {}/{} ({:.1f}%)\n'.format(
         ee_correct, len(test_loader.dataset),
        100. * ee_correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset)),ee_correct / float(len(test_loader.dataset))

def distribute(model,clients):
    """
    input:
        model:network element model (NE)
        clients:a list of clients 
    func:
        distribute input model to every client
    """
    pass

def offline_run(args,NE=None):
    # jointly train whole network 
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    if args.offline_dataset in dataset_name.keys():
        num_class=dataset_class_num[args.offline_dataset]
        train_loader = torch.utils.data.DataLoader(
            dataset_name[args.offline_dataset]('./dataset/'+args.offline_dataset,
                                            train=True, download=True,
                                            transform=transforms.ToTensor()),
            batch_size=args.offline_batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            dataset_name[args.offline_dataset]('./dataset/'+args.offline_dataset, 
                                            train=False, transform=transforms.ToTensor()),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        raise ValueError("invalid dataset")
    if NE==None:
        model=model_name[args.model](num_class)
    else:
        model=model_name[args.model](num_class,NE.cfg)
        model.NE=NE
    if args.cuda:
        model.cuda()
    
    optimizer = optim.SGD(model.parameters(), lr=args.offline_lr, momentum=args.offline_momentum, weight_decay=args.offline_weight_decay)
    best_prec1 = 0
    # make save dir if not exists
    if args.save and not os.path.exists(args.save):
        os.makedirs(args.save)
    min_heap=[0]
    heapq.heapify(min_heap)
    for epoch in range(args.offline_epoch):  
        if epoch in [args.offline_epoch*0.5, args.offline_epoch*0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        offline_train(epoch,model,args,optimizer,train_loader)
        prec1,prec2 = test(model,args,test_loader)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if len(min_heap)>args.heap_size:#生长策略
            if heapq.heappushpop(min_heap,prec1)==prec1 and prec1>prec2:
                model.grow()   
                min_heap.clear()
        else:
            heapq.heappush(min_heap,prec1)
        if args.cuda:
            model.cuda()
        save_checkpoint({
            'epoch': epoch + 1,
            'nclass':num_class,
            'state_dict': model.state_dict(),
            'NE_state_dict':model.NE.state_dict(),
            'growth':model.growth,
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
            'cfg':model.cfg,
            }, is_best, filepath=args.save_server)  

    print("Best accuracy: "+str(best_prec1))
    return model.NE
    
       

def deploy(args):
    # split network to NE and NG
    net=torch.load(os.path.join(args.save_server,'model_best.pth.tar'))
    model_name[args.model].growth=net['growth']
    model=model_name[args.model](net['nclass'])
    NE=model.NE
    NE.load_state_dict(net["NE_state_dict"])
    NE.cuda()
    NE=prune_channel(NE,args)
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






