import torch
import torch.nn as nn

class SubModelB(nn.Module):
    def __init__(self):
        super(SubModelB, self).__init__()
        self.layer = nn.Linear(10, 20)

    def forward(self, x):
        return self.layer(x)

class ModelA(nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        self.sub_model_b = SubModelB()
        
        # 在这里给子模型B添加一个类方法
        def new_method(self, x):
            # print("iho")
            return self.layer(x) * 2
        
        setattr(self.sub_model_b, 'new_method', new_method.__get__(self.sub_model_b))

    def forward(self, x):
        return self.sub_model_b(x)

# 使用示例
model_a = ModelA()
input_tensor = torch.randn(5, 10)
output = model_a(input_tensor)
print(output)

# 调用子模型B的新方法
output_new_method = model_a.sub_model_b.new_method(input_tensor)
print(output_new_method)
