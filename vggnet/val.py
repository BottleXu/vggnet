


import torch
import cv2
import numpy as np

from PIL import Image


import module_load as mdld

from torchvision import transforms

dataset_dir, pt_dir = mdld.get_path()
# model = mdld.VGG16_NET()
# model.load_state_dict(torch.load(pt_dir))
# model = torch.load(pt_dir)

model = mdld.VGG16_NET()
model.load_state_dict(torch.load(pt_dir))  # `pt_dir` is the path to your saved model
model.eval()  # Set to evaluation mode
device = 'cuda'
model.to('cuda')  # Move to CPU or GPU as appropriate


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# model.eval()


model.eval()


# img = cv2.imread("../test/plain_1.jpg")
# resized_image = cv2.resize(img, (224, 224))

# image = Image.open("../test/plain_1.jpg").convert('RGB')
image = Image.open("../test/deer_1.jpeg").convert('RGB')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)
    _, predicted = output.max(1)  # Gets the class index with the highest score
    predicted_class = predicted.item()


classes_in = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Predicted class: {classes_in[predicted_class]}")
