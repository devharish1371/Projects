from dataset import load_elliptic_bitcoin_data
from model import GNNModel
from train import train_gnn

# File paths
node_file = "elliptic_txs_features.csv"
edge_file = "elliptic_txs_edgelist.csv"
class_file = "elliptic_txs_classes.csv"

# Load data
graph_data = load_elliptic_bitcoin_data(node_file, edge_file, class_file)

# Train GNN
trained_model = train_gnn(graph_data)
