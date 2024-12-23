import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_mnist_data(batch_size=64):
    """
    Loads the MNIST dataset and returns DataLoader objects for training and testing.

    Args:
        batch_size (int): Number of samples per batch.

    Returns:
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the testing dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to mean=0.5, std=0.5
    ])
    
    # Download and load the training dataset
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    
    # Download and load the test dataset
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
