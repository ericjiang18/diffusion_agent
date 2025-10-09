import torch
import torch.nn.functional as F

from .proxy_reward_model import ProxyRewardModel
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse


class GuidedGeneration:
    """
    Manages the MACP-guided generation process using Zeroth-Order Optimization.
    This class is designed to be used as a 'guider' by the diffusion model's sampling process.
    Uses PyG Batch for evaluating candidates with the ProxyRewardModel.
    """
    def __init__(self,
                 proxy_reward_model: ProxyRewardModel,
                 macp_weights: dict,
                 num_candidates_per_step: int = 10,
                 device='cpu'):
        self.proxy_reward_model = proxy_reward_model
        self.proxy_reward_model.eval()
        self.macp_weights = macp_weights
        self.num_candidates_per_step = num_candidates_per_step
        self.device = device

    def _calculate_composite_macp_reward(self, predicted_rewards: torch.Tensor):
        """
        Calculates the composite MACP reward from predicted components.
        Args:
            predicted_rewards (torch.Tensor): (batch_size, num_reward_components)
        Returns:
            torch.Tensor: (batch_size,) composite MACP scores.
        """
        if not hasattr(self, 'macp_weights_tensor_ordered'): # Changed attribute name for clarity
            # Ensure keys in macp_weights match the order of proxy_model's output components
            # This requires a defined convention for proxy_model's output columns.

            # Attempt to use 'reward_component_names' from proxy_model if it exists and is a list
            # This is a more robust way to ensure order.
            weights_list = []
            component_keys_source = "macp_weights.keys()" # Default source for keys

            if hasattr(self.proxy_reward_model, 'reward_component_names') and \
               isinstance(self.proxy_reward_model.reward_component_names, list):
                try:
                    weights_list = [self.macp_weights[key] for key in self.proxy_reward_model.reward_component_names]
                    component_keys_source = "proxy_model.reward_component_names"
                except KeyError as e:
                    raise ValueError(f"Key '{e}' from proxy_model.reward_component_names "
                                     "not found in provided macp_weights dictionary.")
            else: # Fallback to dict order if attribute not present or not a list
                 print("WARNING: ProxyRewardModel does not have 'reward_component_names' list attribute "
                       "or it's not a list. Relying on macp_weights dict order. "
                       "Ensure this matches proxy output column order.")
                 weights_list = [self.macp_weights[key] for key in self.macp_weights.keys()]

            weights_tensor = torch.tensor(weights_list, device=self.device, dtype=torch.float32)

            if self.proxy_reward_model.num_reward_components != predicted_rewards.shape[1]:
                 # This check is against the passed predicted_rewards, which should match proxy's output dim
                 pass # Already checked proxy_model.num_reward_components during its init.

            if weights_tensor.shape[0] != predicted_rewards.shape[1]:
                raise ValueError(
                    f"Mismatch between number of MACP weights ({weights_tensor.shape[0]}) derived from "
                    f"{component_keys_source} and number of "
                    f"predicted reward components from proxy model ({predicted_rewards.shape[1]}). Ensure consistency."
                )
            self.macp_weights_tensor_ordered = weights_tensor.unsqueeze(0) # Shape: (1, num_reward_components)

        composite_reward = torch.sum(predicted_rewards * self.macp_weights_tensor_ordered, dim=1)
        return composite_reward


    def guide(self,
              current_At_prob: torch.Tensor,
              timestep: torch.Tensor,
              unguided_A0_prediction: torch.Tensor,
              node_features: torch.Tensor, # (batch_size, num_nodes, node_feature_dim)
              task_condition: torch.Tensor  # (batch_size, condition_dim)
             ):
        """
        Performs one step of guided sampling.
        Args:
            node_features: Original node features for the batch of graphs being generated.
            task_condition: Original task condition for the batch.
        Returns:
            torch.Tensor: The A0_best_candidate (binary {0,1}) selected by the guider.
                          Shape: (batch_size, num_nodes, num_nodes)
        """
        batch_size, num_nodes, _ = unguided_A0_prediction.shape
        A0_best_candidates_batch = torch.zeros_like(unguided_A0_prediction)

        for i in range(batch_size): # Process each graph in the generation batch individually
            unguided_A0_pred_item = unguided_A0_prediction[i] # (N, N) probabilities for current graph

            # 1. Candidate Generation: Sample K binary graphs from this item's probabilities
            candidate_A0s_binary_list = []
            for _ in range(self.num_candidates_per_step):
                # Bernoulli sampling from the (N,N) probability matrix
                binary_A0_sample = torch.bernoulli(unguided_A0_pred_item).float()
                candidate_A0s_binary_list.append(binary_A0_sample) # List of K tensors, each (N,N)

            if not candidate_A0s_binary_list:
                A0_best_candidates_batch[i] = unguided_A0_pred_item # Fallback if K=0 or error
                continue

            # 2. Proxy Evaluation using PyG Batch for these K candidates
            pyg_data_candidates = []
            # Node features for the current graph in the generation batch
            node_features_for_current_graph = node_features[i] # Shape: (N, node_feature_dim)
            # Task condition for the current graph (graph-level)
            # Needs to be (1, condition_dim) for each Data object, PyG Batch handles stacking later
            task_condition_for_current_graph = task_condition[i].unsqueeze(0) # Shape: (1, condition_dim)

            for binary_adj_candidate_matrix in candidate_A0s_binary_list: # binary_adj_candidate_matrix is (N,N)
                edge_index_candidate, _ = dense_to_sparse(binary_adj_candidate_matrix.to(self.device))

                # Each Data object gets node_features_for_current_graph and task_condition_for_current_graph
                data_candidate = Data(
                    x=node_features_for_current_graph.clone().to(self.device),
                    edge_index=edge_index_candidate, # Already on device
                    condition=task_condition_for_current_graph.clone().to(self.device) # Graph-level attr
                )
                pyg_data_candidates.append(data_candidate)

            # Create a PyG Batch from the list of K Data candidates
            # This batch will have K graphs in it.
            pyg_batch_for_proxy_eval = Batch.from_data_list(pyg_data_candidates).to(self.device)

            with torch.no_grad():
                # predicted_rewards_for_candidates will be (K, num_reward_components)
                predicted_rewards_for_candidates = self.proxy_reward_model(pyg_batch_for_proxy_eval)

            # 3. Selection (ZO Optimization part)
            # composite_macp_scores will be (K,)
            composite_macp_scores = self._calculate_composite_macp_reward(predicted_rewards_for_candidates)
            best_candidate_idx_in_k_batch = torch.argmax(composite_macp_scores)

            # The best candidate is one of the binary matrices from candidate_A0s_binary_list
            A0_best_candidate_for_item = candidate_A0s_binary_list[best_candidate_idx_in_k_batch] # (N, N), binary
            A0_best_candidates_batch[i] = A0_best_candidate_for_item # Store it for the i-th graph of original batch

        return A0_best_candidates_batch # (original_batch_size, N, N), binary {0,1}

