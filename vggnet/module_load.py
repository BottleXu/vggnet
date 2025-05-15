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
from mpl_toolkits.mplot3d.proj3d import transform
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

class HyperParams():
    def __init__(self):
        self.learning_rate=0.001
        self.num_epochs= 10
        self.batch_size= 128

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

def get_path():
    code_path = os.getcwd()
    main_dir, main_file = os.path.split(code_path)
    dataset_dir = os.path.join(main_dir, 'dataset')
    pt_dir = os.path.join(main_dir, 'pt/cifar_net.pt')
    return  dataset_dir, pt_dir

def GetTransforms(size, is_filp, nor_mean_array, nor_std_array):

    if is_filp:
        return_transform = transforms.Compose([transforms.Resize((size,size)),
                                         transforms.RandomHorizontalFlip(p=0.7),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=nor_mean_array,
                                                              std=nor_std_array)])
    else:
        return_transform = transforms.Compose([transforms.Resize((size, size)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=nor_mean_array,
                                                                    std=nor_std_array)])
    return return_transform

def LoadCIFAR10DataSet(seed, dataset_dir, tranform_train, val_size):
    torch.manual_seed(seed)
    train = torchvision.datasets.CIFAR10(dataset_dir, train=True, download=True, transform=tranform_train)
    val_size = 10000
