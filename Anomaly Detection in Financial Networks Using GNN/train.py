import torch
import torch.optim as optim
import torch.nn as nn

def train(model, data, train_idx, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data, mask_idx):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        acc = (pred[mask_idx] == data.y[mask_idx]).sum() / mask_idx.size(0)
    return acc.item()

def train_gnn(graph_data, hidden_dim=64, epochs=50, learning_rate=0.01):
    input_dim = graph_data.num_features
    output_dim = graph_data.y.max().item() + 1
    model = GNNModel(input_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train-validation-test split
    num_nodes = graph_data.num_nodes
    train_idx = torch.arange(int(num_nodes * 0.6))  # 60% training
    val_idx = torch.arange(int(num_nodes * 0.6), int(num_nodes * 0.8))  # 20% validation
    test_idx = torch.arange(int(num_nodes * 0.8), num_nodes)  # 20% test

    print("Starting training...")
    for epoch in range(epochs):
        loss = train(model, graph_data, train_idx, optimizer, criterion)
        val_acc = test(model, graph_data, val_idx)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    test_acc = test(model, graph_data, test_idx)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    return model
