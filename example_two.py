import torch
import torch.nn as nn
import torch._dynamo as dynamo
from torch.fx import GraphModule
from torch.fx.passes.graph_drawer import FxGraphDrawer
from torch._dynamo.testing import rand_strided
from random import randint
import types
from tools import inspect_backend

import logging
# torch._logging.set_logs(graph_breaks=True)
import os
os.environ["TORCH_LOGS"] = "+dynamo"

# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.verbose = True
# dynamo.config.debug = True
# torch._dynamo.config.log_level = logging.DEBUG

# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
#         self.conv2 = nn.Conv2d(3, 8, 5, padding=2)
#         self.fc = nn.Linear(16 * 32 * 32, 10)

#     def forward(self, x):
#         y = self.conv1(x)
#         z = self.conv2(x)
#         x = torch.cat([y, z], dim=1)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        path1 = torch.sin(x)
        path2 = torch.cos(x)
        return path1 ** 2 + path2 ** 2

# Part 1: A model that works well with both FX and Dynamo
class TracableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 32 * 32, 10)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Part 2: A model with control flow that breaks FX tracing but works with Dynamo
class ModelWithControlFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 32 * 32, 10)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        
        # Dynamic control flow based on input - Dynamo can handle this!
        if x.sum() > 0:
            x = self.relu(x)
        else:
            x = -x
            
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Part 3: A model with Python operations that challenges tracing
class ModelWithPythonOps(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 32 * 32, 10)
        
    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)


        x = self.relu(x)

        # features = []
        # for i in range(x.size(0)):
        #     features.append(x[i].sum().item())
        
        # min_feature = min(features)
        # x = x * min_feature

        if x.shape[2] == 32:
            print("32")
        else:
            print("not 32")
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Create sample input
sample_input = torch.randn(2, 3, 32, 32)

torch._dynamo.reset()

# model = Model()
# compiled = torch.compile(model, backend=inspect_backend)
# val = compiled(sample_input)
# print(val)

# model1 = TracableModel()
# compiled1 = torch.compile(model1, backend=inspect_backend)
# val1 = compiled1(sample_input)
# print(val1)

# model2 = ModelWithControlFlow()
# compiled2 = torch.compile(model2, backend=inspect_backend)
# val2 = compiled2(sample_input)
# print(val2)

# model3 = ModelWithPythonOps()
# compiled3 = torch.compile(model3, backend=inspect_backend)
# val3 = compiled3(sample_input)
# print(val3)

def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    print("woo")
    if b.sum() < 0:
        b = b * -1
    return x * b

a= torch.randn(10)
b = torch.randn(10)

val = torch.compile(toy_example, backend=inspect_backend)(a, b)
print(val)
# explanation = dynamo.explain(toy_example)(torch.randn(10), torch.randn(10))
# print(explanation)



