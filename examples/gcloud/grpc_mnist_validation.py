import math

import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision.transforms import ToTensor


from appfl.misc.data import *

DataSet_name = "MNIST"
num_channel = 1  # 1 if gray, 3 if color
num_classes = 10  # number of the image classes
num_pixel = 28  # image size = (num_pixel, num_pixel)

class CNN(torch.nn.Module):
    def __init__(self, num_channel, num_classes, num_pixel):
        super().__init__()
        self.conv1 = nn.Conv2d(
            num_channel, 32, kernel_size=5, padding=0, stride=1, bias=True
        )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.act = nn.ReLU(inplace=True)

        X = num_pixel
        X = math.floor(1 + (X + 2 * 0 - 1 * (5 - 1) - 1) / 1)
        X = X / 2
        X = math.floor(1 + (X + 2 * 0 - 1 * (5 - 1) - 1) / 1)
        X = X / 2
        X = int(X)

        self.fc1 = nn.Linear(64 * X * X, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.maxpool(x)
        x = self.act(self.conv2(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


# model = CNN(num_channel, num_classes, num_pixel)

""" Data """
test_data_raw = eval("torchvision.datasets." + DataSet_name)(
    f"../datasets/RawData", download=False, train=False, transform=ToTensor()
)

test_data_input = []
test_data_label = []
for idx in range(len(test_data_raw)):
    test_data_input.append(test_data_raw[idx][0].tolist())
    test_data_label.append(test_data_raw[idx][1])

test_dataset = Dataset(
torch.FloatTensor(test_data_input), torch.tensor(test_data_label)
)
dataloader = server_dataloader = DataLoader(
            test_dataset,
            num_workers=0,
            batch_size=64,
            shuffle=False,
        )

""" Model """
device = "cpu"

file = "./resulting_models/MNIST_CNN_Iter_10.pt" 
model = torch.jit.load(file)
model.eval()
loss_fn = torch.nn.CrossEntropyLoss()
 
model.to(device)

test_loss = 0
correct = 0
tmpcnt = 0
tmptotal = 0
with torch.no_grad():
    for img, target in dataloader:
        tmpcnt += 1
        tmptotal += len(target)
        img = img.to(device)
        target = target.to(device)
        output = model(img)                             
        test_loss += loss_fn(output, target).item()                
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()


test_loss = test_loss / tmpcnt
accuracy = 100.0 * correct / tmptotal

print("test_loss=", test_loss, " accuracy=", accuracy)