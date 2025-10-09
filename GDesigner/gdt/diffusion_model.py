import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .graph_transformer import GraphTransformer

def get_timestep_embedding(timesteps, embedding_dim):
    """
    Build sinusoidal embeddings.
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class ConditionalDiscreteGraphDiffusion(nn.Module):
    def __init__(self,
                 denoising_network: GraphTransformer,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 device='cpu'):
        super(ConditionalDiscreteGraphDiffusion, self).__init__()

        self.denoising_network = denoising_network
        self.num_timesteps = num_timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # For q_sample (forward process)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # For q_posterior (used in reverse process for predicting x_0 from x_t)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        # Clipping posterior_variance to avoid division by zero
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance)

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)


    def _extract(self, a, t, x_shape):
        """Extract coefficients at specified timesteps t and reshape to x_shape."""
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    def q_sample(self, A0, t, noise=None):
        """
        Forward process: Add noise to the graph A0 to get At.
        A0 is a binary adjacency matrix (0 or 1).
        We model this by corrupting edges towards an absorbing state (all-zero matrix - empty graph).
        This means we are more likely to flip 1s to 0s than 0s to 1s as t increases.

        A simpler interpretation for discrete diffusion often involves probabilities.
        Let's assume A0 contains probabilities [0,1] for edges.
        The noising process will drive these probabilities towards 0.5 (random noise)
        or a target absorbing state (e.g. all zeros).

        For discrete graph diffusion, the transition kernel q(A_t | A_{t-1})
        can be defined by flipping edges.
        A common approach for discrete state spaces is to use categorical distributions.

        Here, we'll simplify and treat A0 as continuous values between 0 and 1 (probabilities),
        and the noising process will be similar to continuous diffusion, then clamp/binarize.
        This is a common simplification if the underlying denoising network predicts probabilities.

        Args:
            A0 (torch.Tensor): Original clean graph (batch_size, num_nodes, num_nodes), values in {0, 1}.
            t (torch.Tensor): Timesteps (batch_size,).
            noise (torch.Tensor, optional): Pre-sampled noise. If None, sample Gaussian noise.
        Returns:
            torch.Tensor: Noisy graph At.
        """
        if noise is None:
            noise = torch.randn_like(A0, device=self.device) # Noise is towards the mean of the probabilities (0.5)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, A0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, A0.shape)

        # This is the standard DDPM noising formula, assuming A0 values are scaled to [-1, 1]
        # For binary {0,1} or probabilities [0,1], this needs adjustment.
        # Let's assume A0 is probabilities [0,1]. We can scale to [-1,1] for diffusion.
        A0_scaled = 2 * A0 - 1 # Scale from [0,1] to [-1,1]

        noisy_A_scaled = sqrt_alphas_cumprod_t * A0_scaled + sqrt_one_minus_alphas_cumprod_t * noise

        # Scale back to [0,1] and clamp - this results in probabilities for At
        noisy_A_prob = (noisy_A_scaled + 1) / 2
        noisy_A_prob = torch.clamp(noisy_A_prob, 0.0, 1.0)

        return noisy_A_prob # At is a matrix of probabilities

    def predict_A0_from_At(self, At_prob, t, node_features, task_condition):
        """
        Use the denoising network to predict A0 (probabilities) from At (probabilities).
        Args:
            At_prob (torch.Tensor): Noisy graph probabilities (batch_size, num_nodes, num_nodes).
            t (torch.Tensor): Timesteps (batch_size,).
            node_features (torch.Tensor): Node features (batch_size, num_nodes, node_feature_dim).
            task_condition (torch.Tensor): Task and agent context (batch_size, condition_dim).
        Returns:
            torch.Tensor: Predicted A0 probabilities (batch_size, num_nodes, num_nodes).
        """
        time_embedding = get_timestep_embedding(t, self.denoising_network.time_embed_dim).to(self.device)
        # The denoising network takes At (adj_matrix for attention masking) and predicts A0
        # For attention masking, we might want a binarized version of At_prob
        At_binary_for_masking = (At_prob > 0.5).float() # Or use At_prob directly if attention can handle soft weights

        # Add self-loops to the attention mask to ensure stability.
        # This prevents a node from having no valid attention targets.
        num_nodes = At_binary_for_masking.shape[-1]
        self_loops = torch.eye(num_nodes, device=self.device).unsqueeze(0)
        At_binary_for_masking = torch.clamp(At_binary_for_masking + self_loops, 0, 1)

        predicted_A0_prob = self.denoising_network(
            node_features=node_features,
            adj_matrix=At_binary_for_masking, # Using binarized At for attention mask
            task_condition=task_condition,
            time_embedding=time_embedding
        )
        return predicted_A0_prob

    def q_posterior_mean_variance(self, A0_pred_prob, At_prob, t):
        """
        Compute the mean and variance of the posterior distribution q(A_{t-1} | A_t, A0_pred).
        This is for the reverse process.
        Args:
            A0_pred_prob (torch.Tensor): Predicted clean graph probabilities (batch_size, num_nodes, num_nodes).
            At_prob (torch.Tensor): Noisy graph probabilities at step t (batch_size, num_nodes, num_nodes).
            t (torch.Tensor): Timesteps (batch_size,).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Posterior mean, posterior log variance.
        """
        # Scale probabilities to [-1, 1] for DDPM formulas
        A0_pred_scaled = 2 * A0_pred_prob - 1
        At_scaled = 2 * At_prob - 1

        posterior_mean_coef1_t = self._extract(self.posterior_mean_coef1, t, At_scaled.shape)
        posterior_mean_coef2_t = self._extract(self.posterior_mean_coef2, t, At_scaled.shape)

        posterior_mean_scaled = posterior_mean_coef1_t * A0_pred_scaled + posterior_mean_coef2_t * At_scaled

        # Scale back to [0,1]
        posterior_mean_prob = (posterior_mean_scaled + 1) / 2
        posterior_mean_prob = torch.clamp(posterior_mean_prob, 0.0, 1.0)

        posterior_log_variance_t = self._extract(self.posterior_log_variance_clipped, t, At_scaled.shape)
        return posterior_mean_prob, posterior_log_variance_t

    @torch.no_grad()
    def p_sample(self, At_prob, t, node_features, task_condition, guidance_scale_A0_pred=None):
        """
        Sample A_{t-1} from A_t using the reverse process.
        Args:
            At_prob (torch.Tensor): Current noisy graph probabilities (batch_size, num_nodes, num_nodes).
            t (torch.Tensor): Current timesteps (batch_size,).
            node_features (torch.Tensor): Node features.
            task_condition (torch.Tensor): Task and agent context.
            guidance_scale_A0_pred (torch.Tensor, optional): If provided by an external guider,
                                                           this is the A0 prediction to be used.
        Returns:
            torch.Tensor: Sampled graph probabilities A_{t-1}.
        """
        A0_pred_prob = self.predict_A0_from_At(At_prob, t, node_features, task_condition)
        if guidance_scale_A0_pred is not None:
            # This is where guidance can be injected by overriding/adjusting A0_pred_prob
            # For example, A0_pred_prob = A0_pred_prob + guidance_scale * (guided_A0_target - A0_pred_prob)
            # Or simply replace it if the guider provides a full A0_best candidate
            A0_pred_prob = guidance_scale_A0_pred


        posterior_mean_prob, posterior_log_variance = self.q_posterior_mean_variance(A0_pred_prob, At_prob, t)

        noise = torch.randn_like(At_prob, device=self.device)
        # No noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(At_prob.shape) - 1)))

        # Sample from the posterior (which is Gaussian in the scaled [-1,1] space)
        # Convert mean to scaled space, add scaled noise, then convert back
        posterior_mean_scaled = 2 * posterior_mean_prob - 1
        # The variance is for the scaled space. Noise should be scaled by sqrt(variance).
        sample_scaled = posterior_mean_scaled + nonzero_mask * torch.exp(0.5 * posterior_log_variance) * noise

        sample_prob = (sample_scaled + 1) / 2
        sample_prob = torch.clamp(sample_prob, 0.0, 1.0)

        return sample_prob

    @torch.no_grad()
    def sample(self, num_nodes, batch_size, node_features, task_condition, guider=None):
        """
        Main sampling loop to generate graphs from noise.
        Starts from pure noise At (t=T) and iteratively denoises to A0 (t=0).
        """
        # Start from pure random noise, which in the [0,1] probability space
        # is a uniform distribution. We can sample from it.
        At_prob = torch.rand(batch_size, num_nodes, num_nodes, device=self.device)
        At_prob = (At_prob + At_prob.transpose(-1, -2)) / 2 # Ensure symmetry

        print(f"Sampling {batch_size} graphs, starting from random noise...")
        for t in reversed(range(self.num_timesteps)):
            # Create a tensor of the current timestep for the whole batch
            timesteps = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            guided_A0_override = None
            if guider is not None:
                # Guider needs to predict the unguided A0 first
                unguided_A0_pred = self.predict_A0_from_At(At_prob, timesteps, node_features, task_condition)
                # Then, the guider uses this prediction to find the best guided A0
                guided_A0_override = guider.guide(
                    current_At_prob=At_prob,
                    timestep=timesteps,
                    unguided_A0_prediction=unguided_A0_pred,
                    node_features=node_features,
                    task_condition=task_condition
                )

            At_prob = self.p_sample(
                At_prob=At_prob,
                t=timesteps,
                node_features=node_features,
                task_condition=task_condition,
                guidance_scale_A0_pred=guided_A0_override # Pass the guided prediction
            )
        print("Sampling finished.")
        # The final result is the denoised graph probability at t=0
        return At_prob

    def forward(self, A0_truth_binary, node_features, task_condition):
        """
        The training objective for the diffusion model.
        Calculates the loss for a batch of clean graphs.
        """
        batch_size = A0_truth_binary.shape[0]

        # 1. Sample a random timestep t for each graph in the batch
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()

        # 2. Create the noisy graph At using the forward process (q_sample)
        # Note: A0_truth_binary should be float for q_sample
        At_prob = self.q_sample(A0_truth_binary.float(), t)

        # 3. Use the denoising network to predict the original A0 from the noisy At
        predicted_A0_prob = self.predict_A0_from_At(At_prob, t, node_features, task_condition)

        # 4. Calculate the loss between the predicted A0 and the true A0.
        # We use Binary Cross-Entropy with Logits for numerical stability.
        # The denoising network outputs probabilities (via sigmoid), so we can use BCE loss.
        loss = F.binary_cross_entropy(predicted_A0_prob, A0_truth_binary.float())

        return loss

