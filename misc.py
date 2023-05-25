import torch
from torch import nn
import math

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerClassifier, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(input_dim, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.classifier(output[:, -1, :])
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import LabelEncoder

# 读取文件夹中的所有CSV文件
folder_name = 'dataset/CIC-IDS2017/TrafficLabelling'
files = [f for f in os.listdir(folder_name) if f.endswith('.csv')]

# 读取每个CSV文件并连接在一起
dataframes = []
for file in files:
    file_path = os.path.join(folder_name, file)
    df = pd.read_csv(file_path)
    dataframes.append(df)
data = pd.concat(dataframes, ignore_index=True)

# 将特征和标签分开
X = data.iloc[1:, 8:-1].values  


Y = data.iloc[1:, -1].values  # 取第一行后所有行，最后一列数据作为标签
# 使用 LabelEncoder 将字符串标签转换为整数
encoder = LabelEncoder()
Y = encoder.fit_transform(Y)
# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 将Numpy数组转为PyTorch张量
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.int64))
Y_test = torch.from_numpy(Y_test.astype(np.int64))

# 创建Tensor数据集
train_data = TensorDataset(X_train, Y_train)
test_data = TensorDataset(X_test, Y_test)

# 定义数据加载器
batch_size = 32
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

from torch import optim

# 初始化模型
model = TransformerClassifier(input_dim=76, num_classes=5, nhead=2, nhid=32, nlayers=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs=10
# 训练模型
for epoch in range(epochs):
    print(f"epoch:{epoch}")
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"loss:{loss}")

# 测试模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_dataloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy: {} %'.format(100 * correct / total))
