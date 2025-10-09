import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Layer.
    """
    def __init__(self, condition_dim, feature_dim):
        super(FiLMLayer, self).__init__()
        self.condition_dim = condition_dim
        self.feature_dim = feature_dim
        self.mapper = nn.Linear(condition_dim, 2 * feature_dim)

    def forward(self, x, condition):
        """
        Args:
            x (torch.Tensor): Input features (batch_size, num_nodes, feature_dim).
            condition (torch.Tensor): Conditioning vector (batch_size, condition_dim).
        Returns:
            torch.Tensor: Modulated features.
        """
        gamma_beta = self.mapper(condition)  # (batch_size, 2 * feature_dim)
        gamma = gamma_beta[:, :self.feature_dim].unsqueeze(1)  # (batch_size, 1, feature_dim)
        beta = gamma_beta[:, self.feature_dim:].unsqueeze(1)   # (batch_size, 1, feature_dim)
        return gamma * x + beta

class GraphTransformerLayer(nn.Module):
    """
    A single layer of the Graph Transformer.
    """
    def __init__(self, feature_dim, num_heads, condition_dim, dropout_rate=0.1):
        super(GraphTransformerLayer, self).__init__()
        self.attention = nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout_rate)
        )
        self.film = FiLMLayer(condition_dim, feature_dim)

    def forward(self, x, condition, adj_matrix=None):
        """
        Args:
            x (torch.Tensor): Node features (batch_size, num_nodes, feature_dim).
            condition (torch.Tensor): Conditioning vector (batch_size, condition_dim).
            adj_matrix (torch.Tensor, optional): Adjacency matrix for attention masking
                                                 (batch_size, num_nodes, num_nodes).
                                                 If None, standard self-attention is used.
        Returns:
            torch.Tensor: Output node features.
        """
        # Apply FiLM conditioning
        x_conditioned = self.film(x, condition)

        # Multi-head attention
        attn_mask = None
        if adj_matrix is not None:
            # Create a boolean mask from the adjacency matrix.
            # True values indicate positions that should NOT be attended to.
            attn_mask = ~adj_matrix.bool()
            # Expand mask to match multi-head attention's expectation of (N*num_heads, L, S)
            num_heads = self.attention.num_heads
            attn_mask = attn_mask.repeat_interleave(num_heads, dim=0)

        attn_output, _ = self.attention(x_conditioned, x_conditioned, x_conditioned, attn_mask=attn_mask, need_weights=False)
        x = self.norm1(x_conditioned + attn_output) # Add & Norm

        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output) # Add & Norm
        return x

class GraphTransformer(nn.Module):
    """
    Graph Transformer model for graph-level predictions.
    """
    def __init__(self, task_cond_input_dim: int, node_feature_dim: int, condition_dim: int, time_embed_dim: int, num_layers: int, num_heads: int, output_dim: int, dropout_rate=0.1):
        super(GraphTransformer, self).__init__()
        self.node_feature_dim = node_feature_dim
        self.condition_dim = condition_dim
        self.time_embed_dim = time_embed_dim

        # Embedding for node features (if they are categorical or need projection)
        # For simplicity, assuming node_feature_dim is already the desired embedding size.
        # self.node_embedding = nn.Linear(raw_node_feature_dim, node_feature_dim)

        # Projection layers for conditions
        self.task_cond_projection = nn.Linear(task_cond_input_dim, condition_dim)
        self.time_projection = nn.Linear(time_embed_dim, condition_dim) # Project time_embed to condition_dim for easier fusion

        self.transformer_layers = nn.ModuleList([
            GraphTransformerLayer(node_feature_dim, num_heads, condition_dim + condition_dim, dropout_rate) # condition_dim for task/agent + condition_dim for time
            for _ in range(num_layers)
        ])

        # Output layer to predict edge probabilities
        # Predicts for each pair of nodes (i,j) if an edge exists.
        # Input to MLP is concatenation of final representations of node i and node j.
        self.output_mlp = nn.Sequential(
            nn.Linear(node_feature_dim * 2, node_feature_dim),
            nn.ReLU(),
            nn.Linear(node_feature_dim, 1)
        )

    def forward(self, node_features, adj_matrix, task_condition, time_embedding):
        """
        Args:
            node_features (torch.Tensor): Initial node features (batch_size, num_nodes, node_feature_dim).
            adj_matrix (torch.Tensor): Noisy adjacency matrix (batch_size, num_nodes, num_nodes).
                                       Used for masking in attention.
            task_condition (torch.Tensor): Task and agent team context embedding (batch_size, task_cond_input_dim).
            time_embedding (torch.Tensor): Timestep embedding (batch_size, time_embed_dim).
        Returns:
            torch.Tensor: Predicted edge probability matrix (batch_size, num_nodes, num_nodes).
        """
        # Project conditions to the model's internal dimension
        projected_task_condition = self.task_cond_projection(task_condition)
        projected_time_embedding = self.time_projection(time_embedding)
        combined_condition = torch.cat((projected_task_condition, projected_time_embedding), dim=1)

        x = node_features
        for layer in self.transformer_layers:
            x = layer(x, combined_condition, adj_matrix=adj_matrix)

        # Predict edge probabilities
        batch_size, num_nodes, _ = x.shape
        # Create pairs of node representations
        x_i = x.unsqueeze(2).repeat(1, 1, num_nodes, 1)  # (batch_size, num_nodes, num_nodes, feature_dim)
        x_j = x.unsqueeze(1).repeat(1, num_nodes, 1, 1)  # (batch_size, num_nodes, num_nodes, feature_dim)

        edge_pair_features = torch.cat((x_i, x_j), dim=-1) # (batch_size, num_nodes, num_nodes, feature_dim * 2)

        # Flatten for MLP
        edge_pair_features_flat = edge_pair_features.view(batch_size * num_nodes * num_nodes, -1)
        edge_logits_flat = self.output_mlp(edge_pair_features_flat)

        edge_logits = edge_logits_flat.view(batch_size, num_nodes, num_nodes)

        return torch.sigmoid(edge_logits) # Output probabilities
