from .basic_server import SyncServerHandler
from .basic_client import SGDClientTrainer, SGDSerialClientTrainer
from ...utils.aggregator import Aggregators
from ...utils.serialization import SerializationTool
from ...utils import Logger
import torch
import os
from util.name_match import dataset_class_num,model_name
from torch.utils.data import DataLoader,Dataset
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
##################
#
#      Server
#
##################


class SelfGrowServerHandler(SyncServerHandler):
    """Self Grow server handler."""
    def __init__(self, model: torch.nn.Module, global_round: int, sample_ratio: float, cuda: bool = True, device: str = None, logger: Logger = None, args=None):
        super(SelfGrowServerHandler, self).__init__(model, global_round,sample_ratio,cuda, device,logger)
        net=torch.load(os.path.join(os.getcwd(),f'result/server/model_best.pth.tar')) #path change
        # model_name[args.model].growth=net['growth']
        _model=model_name[args.model](nclasses=dataset_class_num[args.offline_dataset],growth=net['growth'],cfg=net['cfg']) 
        
        _model.load_state_dict(net['state_dict'])
        self.NG =_model.NG
    
    def evaluate(self,args):
         ### add
        kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

        print('server loading')
        
        if os.path.exists(os.path.join(args.save_server,'federated_input_target_0.npy')):
            federated_input_target = np.load(os.path.join(args.save_server,'federated_input_target_0.npy'))
            federated_input = np.load(os.path.join(args.save_server,'federated_input_0.npy'))
            print('server data loaded')
        else:
            print("file missing")
            return
        data = Data.TensorDataset(torch.tensor(federated_input),torch.tensor(federated_input_target))

        federated_loader = DataLoader(dataset=data,batch_size=args.test_batch_size,shuffle=True,**kwargs)
        # print(self._model)
        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # model.load_state_dict(net["state_dict"])
        models = self.NG
        models.to(self._device)
        models.train()
        optimizer = optim.SGD(models.parameters(), 
                                lr=args.online_lr, momentum=args.online_momentum, weight_decay=args.online_weight_decay)
        # print(models)
        counter=0
        counter_data_num = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(federated_loader):
            print('data shape:',data.shape)
            print('target shape:',target.shape)
            data, target = data.to(self._device), target.to(self._device)
            counter_data_num += len(data)
            optimizer.zero_grad()
            output = models(data)
            print('out shape:',output.shape)
            loss = F.cross_entropy(output, target)  # sum up batch loss
            loss.backward()
            optimizer.step()
            # get the index of the max log-probability
            pred=output.data.max(1, keepdim=True)[1]
            # print('pred lengh:',len(pred))
            # print('pred=9 count:',(pred==9).sum())
            counter += len(pred)
            correct += pred.eq(target.view_as(pred)).sum().item()
        print('server increment:  [{}/{} ({:.1f}%)]\tLoss: {:.6f} Accuracy: {}/{} ({:.1f}%)'.format(
                counter_data_num, len(federated_loader.dataset),
                100. * counter_data_num / len(federated_loader.dataset), loss.item(),correct, counter,100. * correct / counter))



##################
#
#      Client
#
##################

