import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset 

from .graph_transformer import GraphTransformer
from .diffusion_model import ConditionalDiscreteGraphDiffusion
from .proxy_reward_model import ProxyRewardModel
from .guided_generation import GuidedGeneration

class GTDFramework:
    """
    Integrates all components of the Guided Topology Diffusion framework:
    - Conditional Discrete Graph Diffusion Model
    - Proxy Reward Model (assumed to be pre-trained for guided generation)
    - Guided Generation process
    """
    def __init__(self,
                 # Diffusion model parameters
                 task_cond_input_dim: int,
                 node_feature_dim: int,
                 condition_dim: int,
                 time_embed_dim: int,
                 gt_num_layers: int,
                 gt_num_heads: int,
                 diffusion_num_timesteps: int = 1000,
                 diffusion_beta_start: float = 0.0001,
                 diffusion_beta_end: float = 0.02,
                 # Proxy model (for guided generation)
                 proxy_reward_model: ProxyRewardModel = None, # Can be None if only training diffusion model
                 # Guidance parameters
                 macp_weights: dict = None, # e.g., {'utility': 1.0, 'cost': -0.1}
                 num_candidates_per_step: int = 10,
                 device='cpu'):

        self.device = device
        self.node_feature_dim = node_feature_dim # Passed to diffusion model for consistency checks

        # 1. Initialize Denoising Network (Graph Transformer)
        self.denoising_network = GraphTransformer(
            task_cond_input_dim=task_cond_input_dim,
            node_feature_dim=node_feature_dim,
            condition_dim=condition_dim,
            time_embed_dim=time_embed_dim,
            num_layers=gt_num_layers,
            num_heads=gt_num_heads,
            output_dim=1 # Not directly used by GT edge predictor, but for consistency
        ).to(self.device)

        # 2. Initialize Conditional Discrete Graph Diffusion Model
        self.diffusion_model = ConditionalDiscreteGraphDiffusion(
            denoising_network=self.denoising_network,
            num_timesteps=diffusion_num_timesteps,
            beta_start=diffusion_beta_start,
            beta_end=diffusion_beta_end,
            device=self.device
        ).to(self.device)

        # 3. Initialize Guider (if proxy model and weights are provided)
        self.guider = None
        if proxy_reward_model is not None and macp_weights is not None:
            self.guider = GuidedGeneration(
                proxy_reward_model=proxy_reward_model.to(self.device), # Ensure proxy is on correct device
                macp_weights=macp_weights,
                num_candidates_per_step=num_candidates_per_step,
                device=self.device
            )
            print("GTDFramework initialized with a Guider for MACP-guided generation.")
        else:
            print("GTDFramework initialized without a Guider (unguided generation or diffusion model training only).")

    def train_diffusion_model(self,
                              dataloader: DataLoader, # Should yield (A0_truth_binary, node_features, task_condition)
                              epochs: int,
                              learning_rate: float = 1e-4):
        """
        Trains the conditional diffusion model (specifically its denoising network).
        """
        optimizer = optim.Adam(self.diffusion_model.parameters(), lr=learning_rate)
        self.diffusion_model.train()

        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (A0_truth_binary, node_features, task_condition) in enumerate(dataloader):
                A0_truth_binary = A0_truth_binary.to(self.device)
                node_features = node_features.to(self.device)
                task_condition = task_condition.to(self.device)

                optimizer.zero_grad()
                loss = self.diffusion_model(A0_truth_binary, node_features, task_condition)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                if batch_idx % 50 == 0: # Log every 50 batches
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

            avg_epoch_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_epoch_loss:.4f}")

        self.diffusion_model.eval() # Set to eval mode after training
        print("Diffusion model training finished.")

    @torch.no_grad()
    def generate_graphs(self,
                        num_graphs: int, # batch_size for generation
                        num_nodes: int,
                        node_features: torch.Tensor, # (num_graphs, num_nodes, node_feature_dim)
                        task_condition: torch.Tensor, # (num_graphs, condition_dim)
                        use_guidance: bool = True):
        """
        Generates a batch of graphs using the diffusion model, optionally with guidance.

        Args:
            num_graphs (int): How many graphs to generate.
            num_nodes (int): Number of nodes for each graph.
            node_features (torch.Tensor): Node features for the graphs.
            task_condition (torch.Tensor): Task/agent conditions for the graphs.
            use_guidance (bool): Whether to use the MACP guider. If True, guider must be initialized.

        Returns:
            torch.Tensor: Generated graph probabilities (num_graphs, num_nodes, num_nodes).
                          Can be binarized by thresholding (e.g., > 0.5).
        """
        if use_guidance and self.guider is None:
            raise ValueError("Guidance requested, but Guider was not initialized (proxy model or MACP weights missing).")

        if node_features.shape[0] != num_graphs or task_condition.shape[0] != num_graphs:
            raise ValueError("Batch size of node_features and task_condition must match num_graphs.")
        if node_features.shape[1] != num_nodes or node_features.shape[2] != self.node_feature_dim:
            raise ValueError(f"Node features shape mismatch. Expected ({num_graphs}, {num_nodes}, {self.node_feature_dim}), "
                             f"got {node_features.shape}")

        self.diffusion_model.eval() # Ensure diffusion model is in eval mode

        active_guider = self.guider if use_guidance else None

        print(f"Starting graph generation for {num_graphs} graphs, each with {num_nodes} nodes...")
        print(f"Using guidance: {use_guidance and active_guider is not None}")

        generated_A0_probs = self.diffusion_model.sample(
            num_nodes=num_nodes,
            batch_size=num_graphs,
            node_features=node_features.to(self.device),
            task_condition=task_condition.to(self.device),
            guider=active_guider
        )
        print("Graph generation finished.")
        return generated_A0_probs

