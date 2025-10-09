import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, Batch # Import PyG Data and Batch

class ProxyRewardModel(nn.Module):
    """
    A GNN-based model to predict proxy rewards for a given graph topology and condition.
    The model predicts multiple reward components (e.g., utility, cost, robustness).
    This version uses PyTorch Geometric Batch object for input.
    """
    def __init__(self,
                 task_cond_input_dim: int,
                 node_feature_dim: int,
                 condition_dim: int,
                 gnn_hidden_dim: int,
                 gnn_layers: int,
                 mlp_hidden_dim: int,
                 num_reward_components: int, # e.g., 3 for utility, cost, robustness
                 dropout_rate=0.1):
        super(ProxyRewardModel, self).__init__()

        self.node_feature_dim = node_feature_dim # This is data.x
        self.condition_dim = condition_dim # This will be on data.condition (custom attr)
        self.num_reward_components = num_reward_components

        # Projection layer for the raw task condition
        self.task_cond_projection = nn.Linear(task_cond_input_dim, condition_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        input_dim = node_feature_dim
        for _ in range(gnn_layers):
            self.gnn_layers.append(GATConv(input_dim, gnn_hidden_dim, heads=4, concat=False, dropout=dropout_rate))
            input_dim = gnn_hidden_dim

        # MLP to process pooled graph embedding and condition
        self.mlp = nn.Sequential(
            nn.Linear(gnn_hidden_dim + condition_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim // 2, num_reward_components)
        )

    def forward(self, pyg_batch: Batch) -> torch.Tensor:
        """
        Forward pass of the proxy reward model using PyG Batch object.

        Args:
            pyg_batch (torch_geometric.data.Batch): A PyG Batch object. Expected attributes:
                - pyg_batch.x (torch.Tensor): Node features (total_num_nodes_in_batch, node_feature_dim).
                - pyg_batch.edge_index (torch.Tensor): Edge indices (2, total_num_edges_in_batch).
                - pyg_batch.batch (torch.Tensor): Batch vector assigning each node to its graph.
                - pyg_batch.condition (torch.Tensor): Condition vectors for each graph in the batch
                                                     (batch_size, condition_dim). This is a custom attribute
                                                     that needs to be added to Data objects before batching.
        Returns:
            torch.Tensor: Predicted reward components (batch_size, num_reward_components).
        """
        x, edge_index, batch_vector = pyg_batch.x, pyg_batch.edge_index, pyg_batch.batch

        # GNN processing
        h = x
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, edge_index)
            h = F.relu(h) # Apply activation after each GNN layer
            # Dropout can also be added here if desired, or within GATConv

        # Global pooling using the batch vector
        # pooled_embeddings: (batch_size, gnn_hidden_dim)
        pooled_embeddings = global_mean_pool(h, batch_vector)

        # Retrieve the condition tensor. It should be (batch_size, condition_dim).
        # The DataLoader should ensure 'condition' is correctly batched (e.g., by repeating or stacking).
        # If 'condition' was added as a graph-level attribute to each Data object,
        # PyG Batch might store it as pyg_batch.condition (if it's a tensor that can be stacked)
        # or it might need to be handled carefully during Data object creation and batching.
        # Let's assume `pyg_batch.condition` is already correctly shaped (batch_size, condition_dim).
        if not hasattr(pyg_batch, 'condition') or pyg_batch.condition is None:
            raise ValueError("pyg_batch must have a 'condition' attribute of shape (batch_size, condition_dim).")

        raw_condition_tensor = pyg_batch.condition
        if raw_condition_tensor.shape[0] != pooled_embeddings.shape[0]:
             # This might happen if condition was not correctly batched.
             # If condition is (total_nodes, cond_dim), we need to select one per graph.
             # A common way is to take the condition of the first node of each graph.
             # This needs to be ensured during Data creation.
             # For now, assume condition_tensor is (batch_size, condition_dim).
            raise ValueError(f"Mismatch in batch size between pooled_embeddings ({pooled_embeddings.shape[0]}) "
                             f"and condition_tensor ({raw_condition_tensor.shape[0]}). "
                             "Ensure 'condition' is correctly batched as a graph-level attribute.")

        # Project the raw condition to the model's internal dimension
        condition_tensor = self.task_cond_projection(raw_condition_tensor)

        # Concatenate pooled graph embedding with the condition vector
        combined_features = torch.cat((pooled_embeddings, condition_tensor), dim=1)

        # Pass through MLP to get reward components
        predicted_rewards = self.mlp(combined_features)

        return predicted_rewards


# Example Training (Conceptual - updated for PyG Batch)
def train_proxy_model_pyg(proxy_model, pyg_dataloader, optimizer, criterion, device):
    proxy_model.train()
    total_loss = 0
    for batch_idx, batch_data in enumerate(pyg_dataloader):
        batch_data = batch_data.to(device)

        # True rewards should be stored as a graph-level attribute on each Data object,
        # e.g., data.y or data.true_rewards. PyG Batch will collate this.
        if not hasattr(batch_data, 'true_rewards') or batch_data.true_rewards is None:
            raise ValueError("BatchData must have 'true_rewards' attribute of shape (batch_size, num_reward_components).")
        true_rewards_batch = batch_data.true_rewards

        optimizer.zero_grad()

        # Forward pass now takes the whole Batch object
        pred_rewards_batch = proxy_model(batch_data)

        loss = criterion(pred_rewards_batch, true_rewards_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 20 == 0: # Log frequency
            print(f"ProxyTrain Batch {batch_idx}/{len(pyg_dataloader)}, Loss: {loss.item():.4f}")

    return total_loss / len(pyg_dataloader)