from statistics import mode
import numpy as np
from torch.utils.data.dataset import Dataset, TensorDataset
import torch
from torch import nn
from torch.utils.data import DataLoader, dataloader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# model
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.conv1 = nn.Conv2d(3,64,(11,11),stride=(4,4),padding=(2,2)) # orignal is stride 4 kernel size 11 
        self.conv2 = nn.Conv2d(64,192,(5,5),stride=(1,1),padding=(2,2))
        self.conv3 = nn.Conv2d(192,384,(3,3),stride=(1,1),padding=(1,1))
        self.conv4 = nn.Conv2d(384,256,(3,3),stride=(1,1),padding=(1,1))
        self.conv5 = nn.Conv2d(256,256,(3,3),stride=(1,1),padding=(1,1))
        
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(256*3*3,4096)
        self.fc2 = nn.Linear(4096,1024)
        self.fc3 = nn.Linear(1024,100)

    def forward(self,x):
        # print(x.shape)
        #=============for web inference=============

        x = x.reshape(128, 128, 4)  
        x = x[:,:,:3]             
        
        x = x.permute(2,0,1)
        
        x = x.reshape(-1,3,128,128)

        # x = F.avg_pool2d(x,10,stride=10)

        x = x / 255

        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        for c in range(3): # every channels
            x = (x-mean[c])/std[c]
           
        
        

        
        
        #=============for web inference=============

        #=============training model================
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool(x)
        # print(x.shape)
        x = x.reshape(-1, 256*3*3)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        # x = F.softmax(x, dim=1)
        return x




from PIL import Image

im = Image.open("/home/fritingo/Documents/FP/Alexnet_web/path/to/cifar100png/train/bed/0003.png")

print(im.size)

im = im.resize((128,128))
rgb = []
for x in range(im.size[0]):
    for y in range(im.size[1]):
        r, g, b = im.getpixel((x,y))
        rgb.append(r)
        rgb.append(g)
        rgb.append(b)
        rgb.append(255)

img_buff = torch.tensor(rgb)
print(img_buff.shape)
print(img_buff)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU state:', device)

net = AlexNet().to(device)
print(net)
net.load_state_dict(torch.load("cifar100_Alexnet.pt"))

net.eval()

img_buff = img_buff.to(device)
output = net(img_buff)

_, predicted = torch.max(output, 1)

classes = ('apple ', 'aquarium_fish ', 'baby ', 'bear ', 'beaver  ', 'bed ', 'bee ', 'beetle ', 'bicycle ', 'bottle ', 'bowl ', 'boy ', 'bridge ', 'bus ', 'butterfly ', 'camel ', 'can ', 'castle ', 'caterpillar ', 'cattle ', 'chair ', 'chimpanzee ', 'clock ', 'cloud ', 'cockroach ', 'couch ', 'cra ', 'crocodile ', 'cup ', 'dinosaur ', 'dolphin ', 'elephant ', 'flatfish ', 'forest ', 'fox ', 'girl ', 'hamster ', 'house ', 'kangaroo ', 'keyboard ', 'lamp ', 'lawn_mower ', 'leopard ', 'lion ', 'lizard ', 'lobster ', 'man ', 'maple_tree ', 'motorcycle ', 'mountain ', 'mouse ', 'mushroom ', 'oak_tree ', 'orange ', 'orchid ', 'otter ', 'palm_tree ', 'pear ', 'pickup_truck ', 'pine_tree ', 'plain ', 'plate ', 'poppy ', 'porcupine ', 'possum ', 'rabbit ', 'raccoon ', 'ray ', 'road ', 'rocket ', 'rose ', 'sea ', 'seal ', 'shark ', 'shrew ', 'skunk ', 'skyscraper ', 'snail ', 'snake ', 'spider ', 'squirrel ', 'streetcar ', 'sunflower ', 'sweet_pepper ', 'table ', 'tank ', 'telephone ', 'television ', 'tiger ', 'tractor ', 'train ', 'trout ', 'tulip ', 'turtle ', 'wardrobe ', 'whale ', 'willow_tree ', 'wolf ', 'woman ', 'worm')

print(classes[predicted])