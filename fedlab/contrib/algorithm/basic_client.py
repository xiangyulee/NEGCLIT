# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import sys
sys.path.append('./')
sys.path.append('.../')
from copy import deepcopy
import torch
from ...core.client.trainer import ClientTrainer, SerialClientTrainer
from ...utils import Logger, SerializationTool
import torch.nn.functional as F
from torch.autograd import Variable
from experiment.SSH_client import sock_client_data


class SGDClientTrainer(ClientTrainer):
    """Client backend handler, this class provides data process method to upper layer.

    Args:
        model (torch.nn.Module): PyTorch model.
        cuda (bool, optional): use GPUs or not. Default: ``False``.
        device (str, optional): Assign model/data to the given GPUs. E.g., 'device:0' or 'device:0,1'. Defaults to None.
        logger (Logger, optional): :object of :class:`Logger`.
    """
    def __init__(self,
                 model:torch.nn.Module,
                 cuda:bool=False,
                 device:str=None,
                 logger:Logger=None):
        super(SGDClientTrainer, self).__init__(model, cuda, device)

        self._LOGGER = Logger() if logger is None else logger

    @property
    def uplink_package(self):
        """Return a tensor list for uploading to server.

            This attribute will be called by client manager.
            Customize it for new algorithms.
        """
        return [self.model_parameters]

    def setup_dataset(self, dataset):
        self.dataset = dataset

    def setup_optim(self, epochs, batch_size, lr):
        """Set up local optimization configuration.

        Args:
            epochs (int): Local epochs.
            batch_size (int): Local batch size. 
            lr (float): Learning rate.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    def local_process(self, args,payload, id):
        model_parameters = payload[0]
        train_loader = self.dataset.get_dataloader(id, self.batch_size,type='train') # change
        test_loader = self.dataset.get_dataloader(id, self.batch_size,type='test') 
        self._LOGGER.info("Local _process run")
        # self.train(model_parameters, train_loader)
        prec1,federated_input,federated_input_target = self.evaluate(model_parameters, test_loader)
            
        if len(federated_input_target)!=0:
            print('federated_input_target saving ')
            save_federated_input_target = np.save(os.path.join(args.save_client,'federated_input_target_{}.npy'.format(id)),
                                                federated_input_target.data.numpy())
            print('federated_input saving ')
            save_federated_input = np.save(os.path.join(args.save_client,'federated_input_{}.npy'.format(id)),
                                    federated_input.data.numpy())
            sock_client_data(args)

    def train(self, model_parameters, train_loader) -> None:
        """Client trains its local model on local dataset.

        Args:
            model_parameters (torch.Tensor): Serialized model parameters.
        """
        SerializationTool.deserialize_model(
            self._model, model_parameters)  # load parameters
        
        self._LOGGER.info("Local train procedure is running")
        for ep in range(self.epochs):
            self._model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                if self.cuda:
                    data, target = data.cuda(self.device), target.cuda(self.device)

                outputs = self._model(data)
                loss = self.criterion(outputs, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if batch_idx % 10 == 0:
                    print('client Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        self._LOGGER.info("Local train procedure is finished")
    def inference(self, model_parameters, test_loader,threshold=500):
        
        SerializationTool.deserialize_model(
            self._model, model_parameters)  # load parameters
        # print('self._model',self._model)
        self._LOGGER.info("Local train procedure is running")
        self._model.cuda()
        self._model.eval()
        pred=[]
        federated_input = []
        inference_input_tensor=[]

        with torch.no_grad():
            for _, (data, _) in enumerate(test_loader):

                data= data.cuda()
                data= Variable(data, volatile=True)
                # print(data.shape)
                output = self._model(data)
                # print(entropy(output))
                if self.entropy(output)> threshold:
                    federated_input.append(self._model.features(data).cpu()) # 合并fearure
   
                else:
                    pred.append(output.data.max(1, keepdim=True)[1]) # get the index of the max log-probability

            if len(federated_input) != 0:
                inference_input_tensor = torch.cat(federated_input,dim=0)

        return inference_input_tensor,pred

    def evaluate(self, model_parameters, test_loader,threshold=0.5):
        """Evaluate quality of local model."""
        """Client evaluates its local model on local dataset.

        Args:
            model_parameters (torch.Tensor): Serialized model parameters.
        """
        SerializationTool.deserialize_model(
            self._model, model_parameters)  # load parameters
        # print('self._model',self._model)
        self._LOGGER.info("Local train procedure is running")
        self._model.cuda()
        self._model.eval()
        test_loss = 0
        correct = 0
        federated_input = []
        federated_input_target = []
        federated_input_tensor=[]
        federated_input_target_tensor=[]
        test_target = []
        counter=0
        # print(self._model)
        with torch.no_grad():
            for data, target in test_loader:

                data, target = data.cuda(), target.cuda()
                # data, target = Variable(data, volatile=True), Variable(target)

                output = self._model(data)
                for i,single_output in enumerate(output):
                    # print( i,' th:','entropy:',self.entropy(single_output))
                    if self.entropy(single_output)> threshold:
                        federated_input.append(self._model.features(data).cpu().numpy()[i]) # 合并fearure
                        federated_input_target.append(target.cpu().numpy()[i])    
                    else:
                        # output = self._model(data)
                        
                        test_target.append(target[i])
                        test_loss += F.cross_entropy(single_output, target[i], reduction='sum').item() # sum up batch loss
                        pred = single_output.data.max(0, keepdim=True)[1] # get the index of the max log-probability
                        counter+=len(pred)
                        correct += pred.eq(target[i].data.view_as(pred)).cpu().sum()

            test_loss /= len(test_loader.dataset)
            print('\nClient Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
                test_loss, correct, counter,100. * correct / counter))
            if len(federated_input_target) != 0:
                federated_input_tensor = torch.tensor(np.array(federated_input))
                federated_input_target_tensor = torch.tensor(federated_input_target)
                # federated_input_tensor = torch.cat(federated_input,dim=0)
                # federated_input_target_tensor = torch.cat(federated_input_target,dim=0)

            # print('target len : ',len(federated_input_target_tensor))
        correct_rate=100. * correct / counter if counter!=0 else 0
        return correct_rate,federated_input_tensor,federated_input_target_tensor

    def entropy(self,x):
        p=F.softmax(x,dim=0)
        p=p.detach().cpu().numpy().reshape(-1)
        Hp = -sum([p[i] * np.log(p[i]) for i in range(len(p))])
        return Hp   



class SGDSerialClientTrainer(SerialClientTrainer):
    """Deprecated
    Train multiple clients in a single process.

    Customize :meth:`_get_dataloader` or :meth:`_train_alone` for specific algorithm design in clients.

    Args:
        model (torch.nn.Module): Model used in this federation.
        num_clients (int): Number of clients in current trainer.
        cuda (bool): Use GPUs or not. Default: ``False``.
        device (str, optional): Assign model/data to the given GPUs. E.g., 'device:0' or 'device:0,1'. Defaults to None.
        logger (Logger, optional): Object of :class:`Logger`.
        personal (bool, optional): If Ture is passed, SerialModelMaintainer will generate the copy of local parameters list and maintain them respectively. These paremeters are indexed by [0, num-1]. Defaults to False.
    """
    def __init__(self, model, num_clients, cuda=False, device=None, logger=None, personal=False) -> None:
        super().__init__(model, num_clients, cuda, device, personal)
        self._LOGGER = Logger() if logger is None else logger
        self.cache = []

    def setup_dataset(self, dataset):
        self.dataset = dataset

    def setup_optim(self, epochs, batch_size, lr):
        """Set up local optimization configuration.

        Args:
            epochs (int): Local epochs.
            batch_size (int): Local batch size. 
            lr (float): Learning rate.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    @property
    def uplink_package(self):
        package = deepcopy(self.cache)
        self.cache = []
        return package

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        for id in id_list:
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            pack = self.train(model_parameters, data_loader)
            self.cache.append(pack)

    def train(self, model_parameters, train_loader):
        """Single round of local training for one client.

        Note:
            Overwrite this method to customize the PyTorch training pipeline.

        Args:
            model_parameters (torch.Tensor): serialized model parameters.
            train_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
        """
        self.set_model(model_parameters)
        self._model.train()

        for _ in range(self.epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return [self.model_parameters]