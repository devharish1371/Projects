import torch
from torch.utils.data import DataLoader
from torch import nn
from dataset import preprocess_dataset
from model import TransformerClassifier
from train import train_model, evaluate_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Step 1: Load the dataset
    train_dataset, test_dataset, tokenizer = preprocess_dataset()
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Step 2: Initialize the model
    model_name = "bert-base-uncased"  # Use GPT or LLaMA-small similarly
    model = TransformerClassifier(model_name=model_name, num_classes=2).to(device)

    # Step 3: Define optimizer and loss function
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    # Step 4: Train the model
    epochs = 3
    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        accuracy = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
