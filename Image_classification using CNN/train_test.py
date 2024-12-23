import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, criterion, optimizer, device):
    """
    Train the CNN model for one epoch.
    
    Args:
    - model: The CNN model.
    - train_loader: DataLoader for training data.
    - criterion: Loss function.
    - optimizer: Optimizer for updating model parameters.
    - device: Device to use (CPU or GPU).
    
    Returns:
    - avg_loss: Average training loss for the epoch.
    """
    model.train()  # Set model to training mode
    total_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Training Loss: {avg_loss:.4f}")
    return avg_loss

def test_model(model, test_loader, criterion, device):
    """
    Evaluate the CNN model on test data.
    
    Args:
    - model: The CNN model.
    - test_loader: DataLoader for test data.
    - criterion: Loss function.
    - device: Device to use (CPU or GPU).
    
    Returns:
    - avg_loss: Average test loss.
    - accuracy: Test accuracy.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy
