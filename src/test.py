import torch
import torch.nn as nn
import torch.optim as optim
import os

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="4,5"

#print(torch.cuda.current_device())
#print(torch.cuda.get_device_name())

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5)

    def forward(self, x):
        x = self.relu(self.net1(x))
        return self.net2(x)

model = ToyModel()
#loss_fn = nn.MSELoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001)

#optimizer.zero_grad()
#outputs = model(torch.randn(20, 10))
#labels = torch.randn(20, 5).to('cuda:1')
#loss_fn(outputs, labels).backward()
#optimizer.step()

if('net1' in dir(model)):
    print(model.net1)
if('net3' in dir(model)):
    print(model.net3)
