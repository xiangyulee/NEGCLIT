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

import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch.utils.data as Data

from .basic_dataset import FedDataset, BaseDataset
from ...utils.data.functional import noniid_slicing, random_slicing


class PathologicalUSTC(FedDataset):
    """The partition stratigy in FedAvg. See http://proceedings.mlr.press/v54/mcmahan17a?ref=https://githubhelp.com

        Args:
            root (str): Path to download raw dataset.
            path (str): Path to save partitioned subdataset.
            num_clients (int): Number of clients.
            shards (int, optional): Sort the dataset by the label, and uniformly partition them into shards. Then 
            download (bool, optional): Download. Defaults to True.
        """
    def __init__(self, root, path, num_clients=100, shards=10, download=False, preprocess=False) -> None:
        self.root = os.path.expanduser(root)
        self.path = path
        self.num_clients = num_clients
        self.shards = shards
        if preprocess:
            self.preprocess(num_clients, shards, download)

    def preprocess(self, download=True):
        # self.num_clients = num_clients
        # self.shards = shards
        self.download = download

        if os.path.exists(self.path) is not True:
            os.mkdir(self.path)
        
        if os.path.exists(os.path.join(self.path, "train")) is not True:
            os.mkdir(os.path.join(self.path, "train"))
            os.mkdir(os.path.join(self.path, "var"))
            os.mkdir(os.path.join(self.path, "test"))
            
        # train
        USTC = DealDataset(self.path,'train-images-idx3-ubyte','train-labels-idx1-ubyte')
        data_indices = noniid_slicing(USTC, self.num_clients, self.shards)

        samples, labels = [], []
        for x, y in USTC:
            samples.append(x)
            labels.append(y)
        for id, indices in data_indices.items():
            data, label = [], []
            for idx in indices:
                x, y = samples[idx], labels[idx]
                data.append(x)
                label.append(y)
            dataset = BaseDataset(data, label)
            torch.save(dataset, os.path.join(self.path, "train", "data{}.pkl".format(id)))

        # test
        USTC_test = DealDataset(self.path,'test-images-idx3-ubyte','test-labels-idx1-ubyte')
        data_indices = noniid_slicing(USTC_test, self.num_clients, self.shards)

        samples, labels = [], []
        for x, y in USTC_test:
            samples.append(x)
            labels.append(y)
        for id, indices in data_indices.items():
            data, label = [], []
            for idx in indices:
                x, y = samples[idx], labels[idx]
                data.append(x)
                label.append(y)
            dataset = BaseDataset(data, label)
            torch.save(dataset, os.path.join(self.path, "test", "data{}.pkl".format(id)))

    def get_dataset(self, id, type="train"):
        """Load subdataset for client with client ID ``cid`` from local file.

        Args:
            cid (int): client id
            type (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.

        Returns:
            Dataset
        """
        dataset = torch.load(os.path.join(self.path, type, "data{}.pkl".format(id)))
        return dataset

    def get_dataloader(self, id, batch_size=None, type="train"):
        """Return dataload for client with client ID ``cid``.

        Args:
            cid (int): client id
            batch_size (int, optional): batch size in DataLoader.
            type (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.
        """
        dataset = self.get_dataset(id, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader


def load_my_data(data_folder, data_name, label_name):
    """
        data_folder: 文件目录
        data_name： 数据文件名
        label_name：标签数据文件名
    """
    with open(os.path.join(data_folder,label_name), 'rb') as lbpath: # rb表示的是读取二进制数据
        # print(lbpath)
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with open(os.path.join(data_folder,data_name), 'rb') as imgpath:
        # print(imgpath)
        x_train = np.frombuffer(
                imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    return (x_train, y_train)
class DealDataset(Dataset):
    """
        读取数据、初始化数据
    """
    def __init__(self, folder, data_name, label_name,transform=None):
        (train_set, train_labels) = load_my_data(folder, data_name, label_name) 
        self.train_set = train_set
        self.targets = train_labels
        self.transform = transform

    def __getitem__(self, index):

        img, targets = self.train_set[index], int(self.targets[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, targets

    def __len__(self):
        return len(self.train_set)  
