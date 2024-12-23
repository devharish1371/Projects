import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_mnist_data
from model import CNN
from train_test import train_model, test_model


def main():
    # Configuration
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Load data
    train_loader, test_loader = load_mnist_data(batch_size=batch_size)

    # Initialize model, loss function, and optimizer
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training and testing loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        test_loss, accuracy = test_model(model, test_loader, criterion, device)

    print("Training completed!")

    # Save the model
    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("Model saved as mnist_cnn.pth.")

if __name__ == "__main__":
    main()
