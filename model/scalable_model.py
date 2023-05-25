import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from stable_baselines3 import PPO,DQN
from stable_baselines3.common.callbacks import EvalCallback,StopTrainingOnNoModelImprovement,StopTrainingOnRewardThreshold,StopTrainingOnMaxEpisodes
from model.resnet import ResNet_E,BasicBlock,Bottleneck
from model.block import SimpleCNN,SEBlockCNN
from model.transformer import Transformer_E

class ScalableNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.search_space=[BasicBlock,Bottleneck]
        self.search_space=[SimpleCNN,SimpleCNN]
    def forward(self,x):
        raise NotImplementedError()
    def grow(self):
        raise NotImplementedError()
    
class ScalableResNetNG(ScalableNetwork):
    
    def __init__(self,in_planes, nclasses ,NE,input_size=[8,8],growth=[],cfg=None):
        super(ScalableResNetNG, self).__init__()
        self.growth=growth
        self.nclasses=nclasses
        self.NG_pred=nn.Sequential(nn.Flatten(),nn.Linear(in_planes*input_size[0]*input_size[1],nclasses))
        self.NG_block=nn.ModuleList()
        for i in range(len(self.growth)):
            best_model_index=self.growth[i][0]
            channels=self.growth[i][1]
            kernel_sizes=self.growth[i][2]
            self.NG_block.append(
            self.search_space[best_model_index](in_planes,channels,kernel_sizes,in_planes,NE,self))
    def grow(self,in_planes,best_model_index,channels,kernel_sizes,NE):
        
        self.NG_block.append(self.search_space[best_model_index](in_planes,channels,kernel_sizes,in_planes,NE,self))
        self.growth.append([])
        self.growth[-1].append(best_model_index)
        self.growth[-1].append(channels)
        self.growth[-1].append(kernel_sizes)
        
    def forward(self, x):
        
        for layer in self.NG_block:
            x = layer(x)
        
        logits=self.NG_pred(x)
        return logits
    
class ScalableTransformerNG(ScalableNetwork):
    
    def __init__(self,in_planes, nclasses ,NE,input_size=[8,8],growth=[],cfg=None):
        super(ScalableResNetNG, self).__init__()
        self.growth=growth
        self.nclasses=nclasses
        self.NG_pred=nn.Sequential(nn.Flatten(),nn.Linear(in_planes*input_size[0]*input_size[1],nclasses))
        self.NG_block=nn.ModuleList()
        for i in range(len(self.growth)):
            best_model_index=self.growth[i][0]
            channels=self.growth[i][1]
            kernel_sizes=self.growth[i][2]
            self.NG_block.append(
            self.search_space[best_model_index](in_planes,channels,kernel_sizes,in_planes,NE,self))
    def grow(self,in_planes,best_model_index,channels,kernel_sizes,NE):
        
        self.NG_block.append(self.search_space[best_model_index](in_planes,channels,kernel_sizes,in_planes,NE,self))
        self.growth.append([])
        self.growth[-1].append(best_model_index)
        self.growth[-1].append(channels)
        self.growth[-1].append(kernel_sizes)
        
    def forward(self, x):
        
        for layer in self.NG_block:
            x = layer(x)
        
        logits=self.NG_pred(x)
        return logits
    
class ScalableResNet(nn.Module):
    def __init__(self, nclasses ,growth=[],cfg=None):
        super(ScalableResNet, self).__init__()
        self.nclasses=nclasses
        self.NE=ResNet_E(nclasses,cfg=cfg)
        self.NG=ScalableResNetNG(self.NE.features_planes,nclasses,self.NE,growth=growth,cfg=cfg)
        self.cfg=cfg
        self.threshold = torch.ones(1)
        

    def grow(self): 
        in_planes=self.NE.features_planes     
        best_model_index,channels,kernel_sizes = self.choose_BestModel()
        self.NG.grow(in_planes,best_model_index,channels,kernel_sizes,self.NE)
    def forward(self, x):
        x = self.NE.features(x)
        x = self.NG(x)
        return x
    def choose_BestModel(self):
        cifar_transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            
        dataset = CIFAR10(root='./dataset/cifar10', train=True, download=True, transform=cifar_transform)  # 加载数据集
        env = CNNOptimizerEnv(dataset,model1=self.NG.search_space[0],model2=self.NG.search_space[1],NE=self.NE,NG=self.NG,in_channels=256, num_classes=self.NE.features_planes)
        model = DQN("MlpPolicy", env, verbose=1)
        stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=2, min_evals=2)
        # stop_callback = StopTrainingOnMaxEpisodes(max_episodes=2)  # 设置寻优的提前退出
        callback = EvalCallback(env, best_model_save_path=None, log_path=None, eval_freq=10, deterministic=True, 
                                render=False,callback_on_new_best=stop_callback)

        model.learn(total_timesteps=1,callback=callback) 

        best_action = model.action_space.sample()  # Initialize with a random action
        best_reward = -1  # Initialize with a low reward
        obs = env.reset()

        for _ in range(1):  # Try several actions to find the best one
            action, _states = model.predict(obs, deterministic=True)
            _, reward, _, _ = env.step(action)
            if reward > best_reward:
                best_reward = reward
                best_action = action

        channels,kernel_sizes = env.decode_action(best_action)
        best_model_index = env.best_model_index
        # print(best_model)
        return best_model_index,channels,kernel_sizes
    
class ScalableTransformer(nn.Module):
    def __init__(self, nclasses ,growth=[],cfg=None):
        super(ScalableResNet, self).__init__()
        self.nclasses=nclasses
        self.NE=ResNet_E(nclasses,cfg=cfg)
        self.NG=ScalableResNetNG(self.NE.features_planes,nclasses,self.NE,growth=growth,cfg=cfg)
        self.cfg=cfg
        self.threshold = torch.ones(1)
        

    def grow(self): 
        in_planes=self.NE.features_planes     
        best_model_index,channels,kernel_sizes = self.choose_BestModel()
        self.NG.grow(in_planes,best_model_index,channels,kernel_sizes,self.NE)
    def forward(self, x):
        x = self.NE.features(x)
        x = self.NG(x)
        return x
    def choose_BestModel(self):
        cifar_transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            
        dataset = CIFAR10(root='./dataset/cifar10', train=True, download=True, transform=cifar_transform)  # 加载数据集
        env = CNNOptimizerEnv(dataset,model1=self.NG.search_space[0],model2=self.NG.search_space[1],NE=self.NE,NG=self.NG,in_channels=256, num_classes=self.NE.features_planes)
        model = DQN("MlpPolicy", env, verbose=1)
        stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=2, min_evals=2)
        # stop_callback = StopTrainingOnMaxEpisodes(max_episodes=2)  # 设置寻优的提前退出
        callback = EvalCallback(env, best_model_save_path=None, log_path=None, eval_freq=10, deterministic=True, 
                                render=False,callback_on_new_best=stop_callback)

        model.learn(total_timesteps=1,callback=callback) 

        best_action = model.action_space.sample()  # Initialize with a random action
        best_reward = -1  # Initialize with a low reward
        obs = env.reset()

        for _ in range(1):  # Try several actions to find the best one
            action, _states = model.predict(obs, deterministic=True)
            _, reward, _, _ = env.step(action)
            if reward > best_reward:
                best_reward = reward
                best_action = action

        channels,kernel_sizes = env.decode_action(best_action)
        best_model_index = env.best_model_index
        # print(best_model)
        return best_model_index,channels,kernel_sizes

class CNNOptimizerEnv(gym.Env):
    def __init__(self,dataset, model1,model2,NE,NG,in_channels, num_classes, max_channels_range=(4, 9), max_kernel_size_range=(3, 3), device=torch.device('cuda')):
        super(CNNOptimizerEnv, self).__init__()

        self.model1 = model1
        self.model2 = model2
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.max_channels_range = max_channels_range
        self.max_kernel_size_range = max_kernel_size_range
        self.device = device
        self.NE=NE
        self.NG=NG
        self.channels_range = max_channels_range[1] - max_channels_range[0] + 1
        self.kernel_sizes_range = max_kernel_size_range[1] - max_kernel_size_range[0] + 1
        self.num_combinations = self.channels_range * self.kernel_sizes_range
        self.action_space = gym.spaces.Discrete(self.num_combinations)


        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)


       

        train_len = int(0.8 * len(dataset))
        val_len = len(dataset) - train_len
        self.train_dataset, self.val_dataset = random_split(dataset, [train_len, val_len])
    
    
    def step(self, action): # 增加模型只需更改此处（如果不改变搜索空间）
        # channels, kernel_sizes = self.decode_action(action)
        # model = SimpleCNN(self.in_channels, channels, kernel_sizes, self.num_classes, device=self.device)
        # reward = self.train_and_evaluate(model)
        channels,kernel_sizes = self.decode_action(action)
        model_simple_cnn = self.model1(self.in_channels, channels, kernel_sizes, self.num_classes,self.NE,self.NG)
        model_seblock_cnn = self.model2(self.in_channels, channels, kernel_sizes, self.num_classes,self.NE,self.NG)
        reward_simple_cnn = self.train_and_evaluate(model_simple_cnn)
        reward_seblock_cnn = self.train_and_evaluate(model_seblock_cnn)
        reward = max(reward_simple_cnn, reward_seblock_cnn)  # choose best model
        if reward == reward_simple_cnn:
            self.best_model_index = 0
        elif reward == reward_seblock_cnn:
            self.best_model_index = 1
        return None, reward, True, {}


    def reset(self):
        return np.array([0], dtype=np.float32)


    def render(self, mode='human'):
        pass

    def decode_action(self, action):
        kernel_size = (action % self.kernel_sizes_range) + self.max_kernel_size_range[0]
        channels = (action // self.channels_range) + self.max_channels_range[0]
        return [channels], [kernel_size]

    def train_and_evaluate(self, model):
        train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=64, shuffle=False)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1):  # Train for a single epoch
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                inputs=self.NE.features(inputs,head=False)
                for layer in self.NG.NG_block:
                    inputs=layer(inputs)
                x=model(inputs)

                outputs = self.NG.NG_pred(x)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs=self.NE.features(inputs,head=False)
                for layer in self.NG.NG_block:
                    inputs=layer(inputs)
                x = model(inputs)
                outputs = self.NG.NG_pred(x)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = correct / total
        # print('epoch:{},accuracy:{}'.format(epoch,accuracy))
        return accuracy



