from sys import argv
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import jpeglib
import numpy as np

PATH = './cifar_net.pth'
IMAGE = 'Cat03.jpeg' if len(argv) < 2 else argv[1]
# batch_size = 1

transform = transforms.Compose([
    torch.from_numpy,
    transforms.Resize((32,32)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# model = torch.load(PATH, weights_only=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))

rgb = jpeglib.read_spatial(IMAGE).spatial

nprgb = np.array([[rgb[:,:,n] for n in range(3)]]).astype(np.float32)
trgb = transform(nprgb)

outputs = net(trgb)

best, predicted = torch.max(outputs, 1)
print(classes[predicted[0]], ' ', predicted[0].item(), ' ', best[0].item())
