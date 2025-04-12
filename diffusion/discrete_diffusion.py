import config
import torch
import sys
import torch.nn.functional as F
from diffusion.diffusion_utils import linear_beta_schedule, cosine_beta_schedule


class DiscreteDiffusion:
    def __init__(self, num_timesteps=config.NUM_TIMESTEPS, vocab_size=config.VOCAB_SIZE, device=config.DEVICE): # Correct ':'
        self.num_timesteps = num_timesteps
        self.vocab_size = vocab_size
        self.device = device

        if config.SCHEDULE_TYPE == 'linear': # Correct ':'
            self.betas = linear_beta_schedule(num_timesteps).to(device)
        elif config.SCHEDULE_TYPE == 'cosine': # Correct ':'
            self.betas = cosine_beta_schedule(num_timesteps).to(device)
        else: # Correct ':'
            raise ValueError(f"Unknown schedule type: {config.SCHEDULE_TYPE}")

        self.alphas = (1. - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(device)

        self.log_q_t_x_t_minus_1 = self._compute_log_q_t_x_t_minus_1()
        self.log_q_t_x_0 = self._compute_log_q_t_x_0()
        self.log_q_t_minus_1_x_t_x_0 = self._compute_log_q_t_minus_1_x_t_x_0()

    def _compute_log_q_t_x_t_minus_1(self): # Correct ':'
        """ Compute log q(x_t | x_{t-1}) """
        log_q = torch.zeros(self.num_timesteps, self.vocab_size, self.vocab_size, device=self.device, dtype=torch.float64)
        eye = torch.eye(self.vocab_size, device=self.device) # Precompute eye
        for t in range(self.num_timesteps): # Correct ':'
            beta_t = self.betas[t]
            diag_indices = torch.arange(self.vocab_size, device=self.device)
            log_q[t, diag_indices, diag_indices] = torch.log(1.0 - beta_t + beta_t / self.vocab_size)
            off_diag_val = torch.log(beta_t / self.vocab_size)
            log_q[t] = log_q[t] + off_diag_val * (1.0 - eye)
        return log_q.float()

    def _compute_log_q_t_x_0(self): # Correct ':'
        """ Compute log q(x_t | x_0) """
        log_q = torch.zeros(self.num_timesteps, self.vocab_size, self.vocab_size, device=self.device, dtype=torch.float64)
        eye = torch.eye(self.vocab_size, device=self.device) # Precompute eye
        for t in range(self.num_timesteps): # Correct ':'
            alpha_bar_t = self.alphas_cumprod[t]
            diag_indices = torch.arange(self.vocab_size, device=self.device)
            log_q[t, diag_indices, diag_indices] = torch.log(alpha_bar_t + (1.0 - alpha_bar_t) / self.vocab_size)
            off_diag_val = torch.log((1.0 - alpha_bar_t) / self.vocab_size)
            log_q[t] = log_q[t] + off_diag_val * (1.0 - eye)
        return log_q.float()

    def _compute_log_q_t_minus_1_x_t_x_0(self): # Correct ':'
        """ Compute log q(x_{t-1} | x_t, x_0) """
        log_q_posterior = torch.zeros(self.num_timesteps, self.vocab_size, self.vocab_size, self.vocab_size, device=self.device, dtype=torch.float64)
        log_q_t_x_t_minus_1_64 = self.log_q_t_x_t_minus_1.double()
        log_q_t_x_0_64 = self.log_q_t_x_0.double()
        for t in range(1, self.num_timesteps): # Correct ':'
            log_q_t_given_t_minus_1 = log_q_t_x_t_minus_1_64[t]
            log_q_t_minus_1_given_0 = log_q_t_x_0_64[t-1]
            log_q_posterior[t] = log_q_t_given_t_minus_1.unsqueeze(1) + log_q_t_minus_1_given_0.unsqueeze(0)
        log_denominator = torch.logsumexp(log_q_posterior, dim=-1, keepdim=True)
        log_denominator = torch.where(torch.isinf(log_denominator), torch.zeros_like(log_denominator), log_denominator)
        log_q_posterior = log_q_posterior - log_denominator
        log_q_posterior = torch.clamp(log_q_posterior, -100.0, 0.0)
        return log_q_posterior.float()

    def q_sample(self, x_start, t): # Correct ':'
        """ Sample x_t given x_0 and t """
        batch_size, seq_len = x_start.shape
        log_q_t_x_0_for_batch_t = self.log_q_t_x_0[t]
        x_start_expanded = x_start.unsqueeze(-1)
        log_q_t_x_0_expanded = log_q_t_x_0_for_batch_t.unsqueeze(1).expand(-1, seq_len, -1, -1)
        x_start_indices = x_start_expanded.unsqueeze(-1).expand(-1, -1, self.vocab_size, -1)
        x_start_indices = torch.clamp(x_start_indices, 0, self.vocab_size - 1)
        log_probs = torch.gather(log_q_t_x_0_expanded, dim=3, index=x_start_indices).squeeze(-1)
        gumbel_noise = torch.rand_like(log_probs)
        gumbel_noise = -torch.log(-torch.log(gumbel_noise.clamp(min=1e-9)) + 1e-9)
        x_t = torch.argmax(log_probs + gumbel_noise, dim=-1)
        return x_t.long()

    def q_posterior_log_probs(self, x_0, x_t, t): # Correct ':'
        """ Compute log q(x_{t-1} | x_t, x_0) """
        batch_size, seq_len = x_0.shape
        log_q_posterior_t = self.log_q_t_minus_1_x_t_x_0[t]
        log_q_posterior_t = log_q_posterior_t.unsqueeze(1).expand(-1, seq_len, -1, -1, -1)
        x_t_idx = x_t.view(batch_size, seq_len, 1, 1, 1).expand(-1, -1, -1, self.vocab_size, self.vocab_size)
        x_t_idx = torch.clamp(x_t_idx, 0, self.vocab_size - 1)
        log_q_posterior_t_i = torch.gather(log_q_posterior_t, dim=2, index=x_t_idx).squeeze(2)
        x_0_idx = x_0.view(batch_size, seq_len, 1, 1).expand(-1, -1, -1, self.vocab_size)
        x_0_idx = torch.clamp(x_0_idx, 0, self.vocab_size - 1)
        log_q_posterior_t_i_j = torch.gather(log_q_posterior_t_i, dim=2, index=x_0_idx).squeeze(2)
        return log_q_posterior_t_i_j

    def p_log_probs(self, model, x_t, t, condition): # Correct ':'
        """ Compute log p_theta(x_0 | x_t, t, condition) """
        log_pred_x0 = model(x_t, t, condition)
        return F.log_softmax(log_pred_x0, dim=-1)

    def p_sample(self, model, x_t, t, condition): # Correct ':'
        """ Sample x_{t-1} from p_theta(x_{t-1} | x_t, t, condition) """
        batch_size, seq_len = x_t.shape
        device = x_t.device
        log_pred_x0 = self.p_log_probs(model, x_t, t, condition)
        log_q_posterior_t = self.log_q_t_minus_1_x_t_x_0[t]
        log_q_posterior_t = log_q_posterior_t.unsqueeze(1).expand(-1, seq_len, -1, -1, -1)
        x_t_idx = x_t.view(batch_size, seq_len, 1, 1, 1).expand(-1, -1, -1, self.vocab_size, self.vocab_size)
        x_t_idx = torch.clamp(x_t_idx, 0, self.vocab_size - 1)
        log_q_posterior_t_i = torch.gather(log_q_posterior_t, dim=2, index=x_t_idx).squeeze(2)
        log_pred_x0_expanded = log_pred_x0.unsqueeze(-1)
        log_sum_terms = log_q_posterior_t_i + log_pred_x0_expanded
        log_p_t_minus_1_given_t = torch.logsumexp(log_sum_terms, dim=2)
        log_p_t_minus_1_given_t = F.log_softmax(log_p_t_minus_1_given_t, dim=-1)
        gumbel_noise = torch.rand_like(log_p_t_minus_1_given_t)
        gumbel_noise = -torch.log(-torch.log(gumbel_noise.clamp(min=1e-9)) + 1e-9)
        x_t_minus_1 = torch.argmax(log_p_t_minus_1_given_t + gumbel_noise, dim=-1)
        return x_t_minus_1.long()

    @torch.no_grad()
    def sample(self, model, condition, shape): # Correct ':'
        """ Generate samples from the model """
        batch_size, seq_len = shape
        device = self.device
        model.eval() # Ensure model is in eval mode for sampling
        x_t = torch.randint(1, self.vocab_size, size=shape, device=device).long() # Avoid sampling PAD initially if possible

        for t in reversed(range(0, self.num_timesteps)): # Correct ':'
            print(f"\rSampling timestep {t+1}/{self.num_timesteps}   ", end="")
            sys.stdout.flush()
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            if t > 0: # Correct ':'
                 x_t = self.p_sample(model, x_t, t_tensor, condition)
            else: # Correct ':'
                 # At t=0, use the model's prediction of x_0 directly
                 log_pred_x0 = self.p_log_probs(model, x_t, t_tensor, condition)
                 # Sample x_0 from the final prediction
                 gumbel_noise = torch.rand_like(log_pred_x0)
                 gumbel_noise = -torch.log(-torch.log(gumbel_noise.clamp(min=1e-9)) + 1e-9)
                 x_t = torch.argmax(log_pred_x0 + gumbel_noise, dim=-1).long()

        print("\nSampling complete.")
        model.train() # Set back to train mode after sampling
        return x_t

    def compute_loss(self, model, x_start, condition, pad_token_id=config.PAD_TOKEN_ID): # Correct ':'
        """ Compute the training loss """
        batch_size, seq_len = x_start.shape
        device = x_start.device
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        x_t = self.q_sample(x_start, t)
        log_pred_x0 = self.p_log_probs(model, x_t, t, condition)
        # Calculate NLL loss
        loss = F.nll_loss(log_pred_x0.permute(0, 2, 1), # Needs [B, K, S] for nll_loss
                          x_start,
                          ignore_index=pad_token_id,
                          reduction='none') # [B, S]
        # Average loss over non-padding tokens
        mask = (x_start != pad_token_id).float()
        loss = (loss * mask).sum() / mask.sum().clamp(min=1) # Average over non-pad tokens per batch
        return loss