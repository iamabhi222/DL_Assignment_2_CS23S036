import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt

seed = 100
torch.manual_seed(seed)
np.random.seed(seed)

# input shape = (3, 224, 224)


# Function to prepare data loaders for training and testing
def prepare_data(augmentation=False, batch_size=64):
    # Define data transformations
    if augmentation:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0, saturation=0, hue=0),
            transforms.RandomRotation(50),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.2), scale=(1, 1.1), shear=0.2),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.Resize(224), # Resize images to 224x224 pixels
            transforms.CenterCrop(224),
            transforms.ToTensor(),   # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images using mean and standard deviation, Mean and SD values for RGB channels
        ])

    test_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load training and testing datasets
    train_data = datasets.ImageFolder(root='./inaturalist_12K/train', transform=train_transforms)
    test_data = datasets.ImageFolder(root='./inaturalist_12K/val', transform=test_transforms)
    
    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    train_data, val_data = random_split(train_data, [train_size, val_size])

    # Create data loaders for training and testing
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader, val_loader

# CNN model class definition
class CNN(nn.Module):
    def __init__(self, n_filters, filter_multiplier, dropout, batch_norm, dense_size, act_func='relu', n_classes=10):
        super(CNN, self).__init__()

        filter_dim = [3, 3, 3, 3, 3] # size of filters in each layer
        
        layers = []

        last_layer_size = 224

        num_filters = []    # number of filters in each layer
        num_filters.append(n_filters)
        for i in range(1, 5):
            num_filters.append(int(num_filters[i - 1] * filter_multiplier))
        # Build the convolutional layers
        for i in range(5):
            last_layer_size = last_layer_size - 2   # size drop due to 2d convolution of stride = 1 and FS = 3
            last_layer_size = last_layer_size // 2  # size drop due to pooling of stride = 2 and filter size = 3
            filter_size = (filter_dim[i], filter_dim[i])
            if i == 0:
                layers.append(nn.Conv2d(3, num_filters[0], filter_size))  # Input channels: 3 (RGB) --> width of input
            else:
                layers.append(nn.Conv2d(num_filters[i - 1], num_filters[i], filter_size))  # Input channels: n_filters
            if batch_norm:
                layers.append(nn.BatchNorm2d(num_filters[i]))  # Batch normalization
            if act_func == 'relu':
                layers.append(nn.ReLU())  # ReLU activation function
            elif act_func == 'leaky':
                layers.append(nn.LeakyReLU(0.3))  # Leaky ReLU activation function
            layers.append(nn.MaxPool2d(2))  # Max pooling Stride = 2

        layers.append(nn.Flatten())  # Flatten layer
        layers.append(nn.Linear(num_filters[4] * last_layer_size * last_layer_size, dense_size))  # Fully connected layer
        layers.append(nn.Dropout(dropout))  # Dropout layer
        if act_func == 'relu':
            layers.append(nn.ReLU())  # ReLU activation function
        elif act_func == 'leaky':
            layers.append(nn.LeakyReLU(0.3))  # Leaky ReLU activation function
        layers.append(nn.Linear(dense_size, n_classes))  # Output layer
        layers.append(nn.Softmax(dim=1))  # Softmax activation function for multiclass classification

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
