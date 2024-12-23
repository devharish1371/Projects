
import pandas as pd
import torch
import networkx as nx
from torch_geometric.utils import from_networkx

def load_elliptic_bitcoin_data(node_file, edge_file, class_file):
    print("Loading node and edge data...")
    
    # Load nodes (features)
    nodes = pd.read_csv(node_file, header=None)
    nodes[0] = nodes[0].astype(str)  # Convert IDs to strings
    features = nodes.iloc[:, 1:-1].values
    node_ids = nodes[0].tolist()

    print(f"Nodes file loaded: {nodes.shape[0]} rows, {nodes.shape[1]} columns.")
    
    # Load edges
    edges = pd.read_csv(edge_file, header=None, names=["source", "target"])
    edges["source"] = edges["source"].astype(str).str.strip()
    edges["target"] = edges["target"].astype(str).str.strip()
    print(f"Edges file loaded: {edges.shape[0]} rows.")
    
    # Filter edges to numeric nodes only
    edges = edges[edges["source"].str.isdigit() & edges["target"].str.isdigit()]
    print(f"Filtered edges count: {edges.shape[0]}")
    
    # Load class labels
    labels = pd.read_csv(class_file, header=None, names=["node_id", "label"], skiprows=1)
    labels["node_id"] = labels["node_id"].astype(str).str.strip()
    label_map = dict(zip(labels["node_id"], labels["label"]))
    print(f"Class file loaded: {labels.shape[0]} rows.")

    # Construct graph
    print("Constructing the graph...")
    G = nx.Graph()
    G.add_edges_from(edges.values)
    
    filtered_features = []
    filtered_labels = []

    # Add node features and labels
    for node, feature_row in zip(node_ids, features):
        if node in G.nodes() and node in label_map:
            G.nodes[node]["x"] = torch.tensor(feature_row, dtype=torch.float32)
            G.nodes[node]["y"] = torch.tensor(int(label_map[node] != "unknown"), dtype=torch.long)
            filtered_features.append(feature_row)
            filtered_labels.append(int(label_map[node] != "unknown"))

    if not filtered_features or not filtered_labels:
        print("No valid nodes found after filtering. Please check the dataset files.")
        return None

    # Convert to PyTorch Geometric format
    print("Converting to PyTorch Geometric format...")
    data = from_networkx(G)
    data.x = torch.tensor(filtered_features, dtype=torch.float32)
    data.y = torch.tensor(filtered_labels, dtype=torch.long)

    print("Elliptic Bitcoin Dataset Loaded:")
    print(data)
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of features: {data.x.shape[1]}")
    print(f"Number of classes: {len(data.y.unique())}")

    return data