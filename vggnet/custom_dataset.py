import os
import yaml
import math
import matplotlib.pylab as plt
import scipy.io
from pytorch_model_summary import summary

from seaborn.utils import DATASET_SOURCE
from torch.onnx.symbolic_opset9 import tensor
from tqdm import tqdm
import re
import cv2
import sys
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, transform
from sklearn.utils import shuffle
from sklearn.utils import resample
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, random_split, Dataset
import torch.optim as optim

# import traffic_signs_datasets_preparing as ts

# path
code_path = os.getcwd()
main_dir, main_file = os.path.split(code_path)
dataset_dir = os.path.join(main_dir, 'dataset/gtsrb')
pt_dir = os.path.join(main_dir, 'pt/fish.pt')

### CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    is_multiple_gpu = True
else:
    is_multiple_gpu = False

### Hyperparameters
# lr=1e-4 #learning_rate = 0.001
lr=1e-4 #learning_rate = 0.001
num_epochs = 50
batch_size = 256
dropout_value = 0.2
is_train =True

class VGG16_NET(nn.Module):
    def __init__(self, data_length, dropout_value):
        super().__init__()
        self.data_length = data_length
        self.dropout_value = dropout_value
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
        self.fc16 = nn.Linear(4096, self.data_length)

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
        x = F.dropout(x, self.dropout_value) #dropout was included to combat overfitting
        x = F.relu(self.fc15(x))
        x = F.dropout(x, self.dropout_value)
        x = self.fc16(x)
        return x

class VGG19_NET(nn.Module):
    def __init__(self, data_length, dropout_value):
        super().__init__()
        self.data_length = data_length
        self.dropout_value = dropout_value
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv14 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv15 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv16 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc17 = nn.Linear(25088, 4096)
        self.fc18 = nn.Linear(4096, 4096)
        self.fc19 = nn.Linear(4096, self.data_length)

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
        x = F.relu(self.conv8(x))
        x = self.maxpool(x)
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.maxpool(x)
        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))
        x = F.relu(self.conv15(x))
        x = F.relu(self.conv16(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc17(x))
        x = F.dropout(x, self.dropout_value) #dropout was included to combat overfitting
        x = F.relu(self.fc18(x))
        x = F.dropout(x, self.dropout_value)
        x = self.fc19(x)
        return x


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class FishDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        # Configure dataset path
        self.dataset_path = os.path.join(root_dir, 'dataset/fish/Fish_Dataset/Fish_Dataset')
        # List to store all image paths
        self.all_path = []

        # Iterate through class folders
        for class_folder in os.listdir(self.dataset_path):
            if class_folder in ['Segmentation_example_script.m', 'README.txt', 'license.txt']:
                continue  # Skip non-class files
            rgb_path = os.path.join(self.dataset_path, class_folder, class_folder)
            if not os.path.exists(rgb_path):
                continue  # Skip if folder doesn’t exist
            all_data = [f for f in os.listdir(rgb_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            # print(f'Found {len(all_data)} images in class "{class_folder}"')
            self.all_path.extend(os.path.join(rgb_path, f) for f in all_data)

        # print(f'\nTotal RGB images found: {len(self.all_path)}')

        self.images_df = pd.DataFrame({
            'Filepath': self.all_path,
            'Label': [os.path.basename(os.path.dirname(path)) for path in self.all_path]
        })

        self.classes = sorted(self.images_df['Label'].unique())
        self.class_to_idx = {label: idx for idx, label in enumerate(self.classes)}
        # self.images_df = self.images_df.sample(frac=1, random_state=42).reset_index(drop=True)

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, idx):
        img_path = self.images_df.iloc[idx, 0]
        label_name = self.images_df.iloc[idx, 1]
        label = self.class_to_idx[label_name]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

    def ImView(self, index):
        img_name = self.images_df.iloc[index, 0]
        image = io.imread(img_name)
        plt.imshow(image)
        plt.show()
class FoodDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode=None):
        self.transform = transform
        # Configure dataset path
        self.dataset_path = os.path.join(root_dir, 'dataset/food-101/food-101/food-101')

        if mode == 'test':
            # self.df = pd.read_csv(os.path.join(self.dataset_path, 'meta/test.txt'), header=None, names=['path'])
            self.df = self.WriteDF(False)
        else:
            # self.df = pd.read_csv(os.path.join(self.dataset_path, 'meta/train.txt'), header = None, names=['path'])
            self.df = self.WriteDF(True)
        print(self.df.head(5))
        self.classes = sorted(self.df['label'].unique())
        self.class_to_idx = {label: idx for idx, label in enumerate(self.classes)}

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        label_name = self.df.iloc[idx, 1]
        label = self.class_to_idx[label_name]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

    def DFSpliter(self, data, class_or_id='id'):
        if class_or_id.upper() == 'CLASS':
            output = data.split('/')[0]

        else:
            output = data.split('/')[-1]
        return output

    def WriteDF(self, is_train)->pd.DataFrame:

        if is_train:
            array = open(os.path.join(self.dataset_path, 'meta/train.txt'), 'r').read().splitlines()
        else:
            array = open(os.path.join(self.dataset_path, 'meta/test.txt'), 'r').read().splitlines()
        # Getting the full path for the images
        img_path = os.path.join(self.dataset_path, 'images')
        full_path = [img_path + img + ".jpg" for img in array]

        # Splitting the image index from the label
        imgs = []
        for img in array:
            img = img.split('/')
            imgs.append(img)

        imgs = np.array(imgs)
        # Converting the array to a data frame
        imgs = pd.DataFrame(imgs[:, 0], imgs[:, 1], columns=['label'])
        # Adding the full path to the data frame
        imgs['path'] = full_path

        # Randomly shuffling the order to the data in the dataframe
        imgs = shuffle(imgs)

        return imgs



    # images_df = cst_data.images_df

tranform_train  = transforms.Compose([transforms.Resize((224,224)),
                                     transforms.RandomHorizontalFlip(p=0.7),
                                     transforms.RandomRotation(20),
                                     transforms.RandomCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])

tranform_val = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])

# prep the train, validation and test dataset
# seed is fixed,
if device == 'cuda':
    torch.cuda.manual_seed(2021)
    # torch.backends.cudnn.deterministic = True
else:
    torch.manual_seed(2021)
# train = torchvision.datasets.
train = FishDataset(main_dir, transform=tranform_train)
test = FishDataset(main_dir, transform=tranform_val)
print(type(train.images_df))
print(train.images_df.shape, test.images_df.shape)

class_len = len(train.classes)
val_size = 1800
train_size = len(train) - val_size
train, val = random_split(train, [train_size, val_size])


#  train, val and test datasets to the dataloader
train_loader = DataLoader(dataset=train,
                          batch_size=batch_size,
                          shuffle=True)

val_loader = DataLoader(dataset=val,
                         batch_size=batch_size,
                         shuffle=False)



model = VGG16_NET(class_len, dropout_value)
# model = VGG19_NET(class_len, 0.5)
# print(summary(model, torch.zeros(1, 3, 224, 224), show_hierarchical=True))

model = model.to(device=device)
if(is_multiple_gpu):
    model = nn.DataParallel(model)
load_model = True
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= lr)
# optimizer = torch.optim.SGD(model.parameters(), lr= lr, momentum=0.9)
# optimizer = torch.optim.RMSprop(model.parameters(), lr= lr)
# optimizer = torch.optim.RMSprop(model.parameters(), lr= lr, momentum=0.9)

# # StepLR: Reduces learning rate every 10 epochs by a factor of 0.1
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# # ExponentialLR with a decay rate of 0.95
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# ReduceLROnPlateau scheduler with validation loss monitoring
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, cooldown=2, threshold=0.01)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
#                                         max_lr=0.01,
#                                         total_steps=None,
#                                         epochs=10,
#                                         steps_per_epoch=len(train_loader))


train_hist = []
val_hist = []
lr_hist = []
patience = 2
best_loss = -1
early_stopping_counter = 0

print(device, is_multiple_gpu, torch.cuda.device_count())
# print(summary(model, torch.zeros((1, 1, 224, 224)), show_input=True))





if is_train:
    for epoch in range(num_epochs):  # I decided to train the model for 50 epochs

        model.train() # 250519 added
        train_loss = 0

        for idx, (images, labels) in tqdm(enumerate(train_loader)):
            train_inputs = images.to(device=device)
            train_labels = labels.to(device=device)
            ## Forward Pass
            optimizer.zero_grad()
            train_outputs = model(train_inputs)
            loss = criterion(train_outputs, train_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # # Step the scheduler after each batch
            # scheduler.step()
        # Step the scheduler at the end of each epoch
        # scheduler.step()

        model.eval() # 250519 added
        val_loss = 0
        with torch.no_grad():
            correct = 0
            samples = 0
            for idx, (images, labels) in tqdm(enumerate(val_loader)):
                val_inputs = images.to(device=device)
                val_labels = labels.to(device=device)
                val_outputs = model(val_inputs)
                loss = criterion(val_outputs, val_labels)
                val_loss = loss.item()

                _, preds = val_outputs.max(1)
                correct += (preds == val_labels).sum()
                samples += preds.size(0)

        train_loss /= len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_hist.append(train_loss)
        val_hist.append(val_loss)
        # lr_hist.append(optimizer.current_learning_rate)


        # # Step the scheduler based on validation loss
        scheduler.step(val_loss)

        accuracy = correct / samples * 100
        print(f"epoch:{epoch + 1},",f"train loss:{train_loss:.4f}, val loss:{val_loss:.4f}, val acc: {accuracy:.2f}%")

        if best_loss == -1 or val_loss < best_loss:
            best_loss = val_loss
            early_stopping_counter = 0
            save_str = 'pt/fish_best_sgd_0.9.pt'
            torch.save(model.state_dict(), os.path.join(main_dir, save_str))
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"best: {epoch + 1 - patience}, early_stopping_counter")
                break

    torch.save(model.state_dict(), os.path.join(main_dir, 'pt/fish_last.pt'))

    plt.plot(train_hist, label='train')
    plt.plot(val_hist, label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
