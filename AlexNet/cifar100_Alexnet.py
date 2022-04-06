from tkinter import X
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# set device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU state:', device)

# transform
transform = transforms.Compose([transforms.ToTensor()]) # 0~255 -> 0~1  //-1~1 , transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

# load data
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
trainLoader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)
testLoader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

# classes
classes = ('apple ', 'aquarium_fish ', 'baby ', 'bear ', 'beaver  ', 'bed ', 'bee ', 'beetle ', 'bicycle ', 'bottle ', 'bowl ', 'boy ', 'bridge ', 'bus ', 'butterfly ', 'camel ', 'can ', 'castle ', 'caterpillar ', 'cattle ', 'chair ', 'chimpanzee ', 'clock ', 'cloud ', 'cockroach ', 'couch ', 'cra ', 'crocodile ', 'cup ', 'dinosaur ', 'dolphin ', 'elephant ', 'flatfish ', 'forest ', 'fox ', 'girl ', 'hamster ', 'house ', 'kangaroo ', 'keyboard ', 'lamp ', 'lawn_mower ', 'leopard ', 'lion ', 'lizard ', 'lobster ', 'man ', 'maple_tree ', 'motorcycle ', 'mountain ', 'mouse ', 'mushroom ', 'oak_tree ', 'orange ', 'orchid ', 'otter ', 'palm_tree ', 'pear ', 'pickup_truck ', 'pine_tree ', 'plain ', 'plate ', 'poppy ', 'porcupine ', 'possum ', 'rabbit ', 'raccoon ', 'ray ', 'road ', 'rocket ', 'rose ', 'sea ', 'seal ', 'shark ', 'shrew ', 'skunk ', 'skyscraper ', 'snail ', 'snake ', 'spider ', 'squirrel ', 'streetcar ', 'sunflower ', 'sweet_pepper ', 'table ', 'tank ', 'telephone ', 'television ', 'tiger ', 'tractor ', 'train ', 'trout ', 'tulip ', 'turtle ', 'wardrobe ', 'whale ', 'willow_tree ', 'wolf ', 'woman ', 'worm')

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
        # input batch size 16 * channel 3 * 32 * 32
        x = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)(x) # 64 * 64
        x = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)(x) # 128 * 128
        # x = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)(x) # 256 * 256
        # print('upsample',x.shape)
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

net = AlexNet().to(device)
print(net)

# set loss optimizer
loss_fnc = nn.CrossEntropyLoss()
lr = 0.001
epochs = 100
optimizer = optim.SGD(net.parameters(),lr=lr,momentum=0.9)

# train
for epoch in range(epochs):
    running_loss = 0.0

    for times, data in enumerate(trainLoader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # print(inputs[0].shape,inputs[0],inputs[0][0])
        # inputs = inputs.view(-1,3*32*32) # for js buffer
        # Zero the parameter gradients
        optimizer.zero_grad()
        # print(inputs.shape)
        # forward + backward + optimize
        outputs = net(inputs)
        loss = loss_fnc(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        if times % 100 == 99 or times+1 == len(trainLoader):
            print('[%d/%d, %d/%d] loss: %.3f' % (epoch+1, epochs, times+1, len(trainLoader), running_loss/times))
    

print('Finished Training')

# Test
correct = 0
total = 0
with torch.no_grad():
    for data in testLoader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # inputs = inputs.view(-1,3*32*32)
        # print(inputs.shape)
        outputs = net(inputs)
        # print(outputs.shape) # 100 classes
        _, predicted = torch.max(outputs.data, 1)
        
        total += 1
        
        correct += (predicted == labels)
        # print(labels,predicted,correct)

print('Accuracy of the network on the 10000 test inputs: %d %%' % (100 * correct / total))

class_correct = list(0. for i in range(100))
class_total = list(0. for i in range(100))
with torch.no_grad():
    for data in testLoader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        
        class_total[int(labels)] += 1
        
        class_correct[int(labels)] += (predicted == labels)
        

for i in range(100):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

torch.save(net.state_dict(), "cifar100_AlexNet.pt")
