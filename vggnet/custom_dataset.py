import os
import yaml
import math
from tqdm import tqdm
import re
import cv2
import sys
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, transform
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
### Hyperparameters
lr=1e-4 #learning_rate = 0.001
num_epochs = 10
batch_size = 128

class VGG16_NET(nn.Module):
    def __init__(self, data_length):
        super().__init__()
        self.data_length = data_length
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

class CustomDataset(Dataset):

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

torch.manual_seed(2021)
# train = torchvision.datasets.
train = CustomDataset(main_dir, transform=tranform_train)
class_len = len(train.classes)
val_size = 1000
train_size = len(train) - val_size
train, val = random_split(train, [train_size, val_size])
test = CustomDataset(main_dir, transform=tranform_val)

#  train, val and test datasets to the dataloader
train_loader = DataLoader(dataset=train,
                          batch_size=batch_size,
                          shuffle=True)

val_loader = DataLoader(dataset=val,
                         batch_size=batch_size,
                         shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device, class_len)
model = VGG16_NET(class_len)
model = model.to(device=device)
load_model = True
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= lr)

for epoch in range(num_epochs):  # I decided to train the model for 50 epochs
    loss_var = 0

    for idx, (images, labels) in tqdm(enumerate(train_loader)):
        images = images.to(device=device)
        labels = labels.to(device=device)
        ## Forward Pass
        optimizer.zero_grad()
        scores = model(images)
        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()
        loss_var += loss.item()
        # if idx % 64 == 0:
        #     print( f'Epoch [{epoch + 1}/{num_epochs}] || Step [{idx + 1}/{len(train_loader)}] || Loss:{loss_var / len(train_loader)}')
    print(f"Loss at epoch {epoch + 1} || {loss_var / len(train_loader)}")

    with torch.no_grad():
        correct = 0
        samples = 0
        for idx, (images, labels) in tqdm(enumerate(val_loader)):
            images = images.to(device=device)
            labels = labels.to(device=device)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum()
            samples += preds.size(0)
        print(
            f"accuracy {float(correct) / float(samples) * 100:.2f} percentage || Correct {correct} out of {samples} samples")


torch.save(model.state_dict(), pt_dir)

