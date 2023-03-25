import torch
import torch.nn as nn

# 假设 NE 和 NG 是两个简单的多层感知器模型
class NE(nn.Module):
    def __init__(self):
        super(NE, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

class NG(nn.Module):
    def __init__(self):
        super(NG, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

ne = NE()
ng = NG()

# 获取 NE 的最后三层
ne_layers = list(ne.model.children())[-3:]

# 获取 NG 的第一层之前的所有层
ng_layers = list(ng.model.children())

# 将 NE 的最后三层插入到 NG 的第一层之前
new_layers = ne_layers + ng_layers

# 创建一个新的模型，将 NE 的最后三层和 NG 的所有层连接起来
connected_model = nn.Sequential(*new_layers)
connected_model2 = nn.Sequential(*list(ne.model.children())[0:-3])
# 测试新模型
input_tensor = torch.randn(1, 10)
output = connected_model(connected_model2(input_tensor))
print(output)
