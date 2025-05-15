### Load and normalize CIFAR10

import os
import yaml
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, random_split
import torch.optim as optim


NUM_CLASSES = 10

class VGG16_NET(nn.Module):
    def __init__(self):
        super(VGG16_NET, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc14 = nn.Linear(25088, 4096)
        self.fc15 = nn.Linear(4096, 4096)
        self.fc16 = nn.Linear(4096, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.maxpool(x)
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.maxpool(x)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc14(x))
        x = F.dropout(x, 0.5) #dropout was included to combat overfitting
        x = F.relu(self.fc15(x))
        x = F.dropout(x, 0.5)
        x = self.fc16(x)
        return x

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


### Hyperparameters
lr=1e-4 #learning_rate = 0.001
num_epochs = 10
batch_size = 128

# print(torch.cuda.is_available(), torch.cuda.device_count())

# Device
# device = torch.device('cuda')

# # path
code_path = os.getcwd()
main_dir, main_file = os.path.split(code_path)
dataset_dir = os.path.join(main_dir, 'dataset')
pt_dir = os.path.join(main_dir, 'pt/cifar_net.pt')
# print(main_dir)
# if not dataset_dir:
#     os.mkdir(dataset_dir, )

# print(dataset_dir)

tranform_train  = transforms.Compose([transforms.Resize((224,224)),
                                     transforms.RandomHorizontalFlip(p=0.7),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])

tranform_test = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])


#prep the train, validation and test dataset

torch.manual_seed(2021)
train = torchvision.datasets.CIFAR10(dataset_dir, train=True, download=True, transform=tranform_train)
val_size = 10000
train_size = len(train) - val_size
train, val = random_split(train, [train_size, val_size])
test = torchvision.datasets.CIFAR10(dataset_dir, train=False, download=True, transform=tranform_test)




#  train, val and test datasets to the dataloader
train_loader = DataLoader(dataset=train,
                          batch_size=batch_size,
                          shuffle=True)

val_loader = DataLoader(dataset=val,
                         batch_size=batch_size,
                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# get some random training images
data_iter = iter(train_loader)
images, labels = next(data_iter)

# show images
# imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

#how the maxpool works while we are preserving the shape in blocks by padding (1,1)
block1 =224
pool1 =math.ceil((block1-3)/2 +1)
print(pool1)


block2=pool1

pool2 =math.ceil((block2-3)/2 +1)
print(pool2)



block3=pool2
pool3 =math.ceil((block3-3)/2 +1)
print(pool3)


block4=pool3
pool4 =math.ceil((block4-3)/2 +1)
print(pool4)


block5=pool4
pool5 =math.ceil((block5-3)/2 +1)
print(pool5)


#After flatten
flatten= pool5 * pool5 * 512
print(f'After flatten:: {flatten}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGG16_NET()
model = model.to(device=device)
load_model = True
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= lr)

for epoch in range(num_epochs):  # I decided to train the model for 50 epochs
    loss_var = 0

    for idx, (images, labels) in enumerate(train_loader):
        images = images.to(device=device)
        labels = labels.to(device=device)
        ## Forward Pass
        optimizer.zero_grad()
        scores = model(images)
        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()
        loss_var += loss.item()
        if idx % 64 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}] || Step [{idx + 1}/{len(train_loader)}] || Loss:{loss_var / len(train_loader)}')
    print(f"Loss at epoch {epoch + 1} || {loss_var / len(train_loader)}")

    with torch.no_grad():
        correct = 0
        samples = 0
        for idx, (images, labels) in enumerate(val_loader):
            images = images.to(device=device)
            labels = labels.to(device=device)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum()
            samples += preds.size(0)
        print(
            f"accuracy {float(correct) / float(samples) * 100:.2f} percentage || Correct {correct} out of {samples} samples")


#SAVES THE TRAINED MODEL
torch.save(model.state_dict(), pt_dir)
