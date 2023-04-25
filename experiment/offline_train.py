import shutil
import torch
import os
import torch.nn as nn
import time
import torch.nn.functional as F
import heapq
from util.name_match import dataset_class_num,dataset_name,model_name
from torchvision import transforms
from torch import optim
import numpy as np
def auto_split(args,NE=None):
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
    model=model_name[args.model](num_class)

    if args.cuda:
        model.cuda()
        for sub_model in model.ee:
            sub_model.cuda()
    
    optimizer = optim.SGD(model.parameters(), lr=args.offline_lr, momentum=args.offline_momentum, weight_decay=args.offline_weight_decay)
    best_prec1 = 0
    # make save dir if not exists
    if args.save and not os.path.exists(args.save):
        os.makedirs(args.save)

    for epoch in range(args.offline_epoch):  
        if epoch in [args.offline_epoch*0.5, args.offline_epoch*0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        offline_train_weighted(epoch,model,args,optimizer,train_loader)
        prec1= test_weighted(model,args,test_loader)
        best_prec1 = max(prec1, best_prec1)
        if args.cuda:
            model.cuda() 
    
    split_layer=model.add_head_split()
    save_checkpoint({
                'nclass':num_class,
                'state_dict': model.state_dict(),
                'NE_state_dict':model.NE.state_dict(),
                'split_layer': split_layer,
                'optimizer': optimizer.state_dict(),
                'cfg':model.cfg,
                },True, filepath=args.save_server)
    
    print("Best accuracy: "+str(best_prec1))
    return model.NE

def fixed_split(args,NE=None):
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

    for epoch in range(args.offline_epoch):  
        if epoch in [args.offline_epoch*0.5, args.offline_epoch*0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        offline_train(epoch,model,args,optimizer,train_loader)
        prec1,prec2 = test(model,args,test_loader)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if args.cuda:
            model.cuda() 
        save_checkpoint({
                'epoch': epoch + 1,
                'nclass':num_class,
                'state_dict': model.state_dict(),
                'NE_state_dict':model.NE.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
                'cfg':model.cfg,
                }, is_best, filepath=args.save_server)

    print("Best accuracy: "+str(best_prec1))
    return model.NE

def self_grow(args,NE=None):
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
        prec1,prec2 = testWithThreshold(model,args,test_loader)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        # model.grow()
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
                'growth':model.NG.growth,
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
                'cfg':model.cfg,
            }, is_best, filepath=args.save_server) 


    print("Best accuracy: "+str(best_prec1))
    return model.NE


def save_checkpoint(state, is_best, filepath): 
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

def updateBN(model,s):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if m.weight.grad!=None:
                m.weight.grad.data.add_(s*torch.sign(m.weight.data))  # L1

def offline_train_weighted(epoch,model,args,optimizer,train_loader):
    start=time.time()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        
        optimizer.zero_grad()
        output = model(data)
        ee_output=model.early_exit(data)
        loss = F.cross_entropy(output, target)
        for o in ee_output:
            loss+=F.cross_entropy(o, target)*0.2
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

def offline_train(epoch,model,args,optimizer,train_loader):
    start=time.time()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        
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


def test_weighted(model,args,test_loader):
    model.eval()
    
    test_loss = 0
    correct = 0
    ee_correct=[0]*model.num_heads
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        ee_output=model.early_exit(data)
        ee_pred=[]
        # test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        for out in ee_output:
            ee_pred.append(out.data.max(1, keepdim=True)[1]) 
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        for i,pre in enumerate(ee_pred):
            ee_correct[i]+= pre.eq(target.data.view_as(pre)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set:Final Average Accuracy: {}/{} ({:.1f}%)\n'.format(
         correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    for i in range(len(ee_correct)):
        print('\nTest set:Layer {} Early Exit Average Accuracy: {}/{} ({:.1f}%)\n'.format(i,
         ee_correct[i], len(test_loader.dataset),
        100. * ee_correct[i] / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))


def test(model,args,test_loader):
    model.eval()
    
    test_loss = 0
    correct = 0
    ee_correct=0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
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

def testWithThreshold(model,args,test_loader):
    model.eval()
    def entropy(x):
        x= torch.softmax(x, dim=1)
        return -torch.sum(x * torch.log2(x + 1e-9), dim=1) 
    test_loss = 0
    correct = 0
    ee_correct=0
    ee_sample=0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        ee_output=model.NE(data)
        en=entropy(ee_output)
        model.threshold=model.threshold.cuda()
        mask=entropy(ee_output) < model.threshold*500
        
        ee_pred= ee_output.data.max(1, keepdim=True)[1]
             
        selected_ee= ee_pred[mask]
        ee_correct+= selected_ee.eq(target[mask].data.view_as(selected_ee)).cpu().sum() 
        ee_sample+=len(selected_ee)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        if(ee_correct /ee_sample<0.9) and model.threshold>0.01:
            
            model.threshold=model.threshold-0.001
        elif (ee_correct /ee_sample>0.95):
            model.threshold=model.threshold+0.001
    # model.threshold.requires_grad=True
    # threshold_loss=(ee_correct /ee_sample-0.9)**2
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    # threshold_loss.backward()
    # optimizer.step()
    # optimizer.zero_grad()

    print('\nTest set:Final Average Accuracy: {}/{} ({:.1f}%)\n'.format(
         correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('\nTest set:Early Exit Average Accuracy: {}/{} ({:.1f}%)\n'.format(
         ee_correct, ee_sample,
        100. * ee_correct /ee_sample))
    print(f'\nCurrent Threshold: {model.threshold}\n')
    return correct / float(len(test_loader.dataset)),ee_correct / ee_sample

