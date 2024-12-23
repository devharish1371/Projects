import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    A Convolutional Neural Network (CNN) for image classification on the MNIST dataset.
    """
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Input channels: 1, Output channels: 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Input channels: 32, Output channels: 64
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 max pooling
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 64 feature maps of size 7x7 flattened
        self.fc2 = nn.Linear(128, 10)  # 10 classes (digits 0-9)

    def forward(self, x):
        # Apply convolution, ReLU, and pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Output layer (no activation, softmax is in loss function)
        return x
