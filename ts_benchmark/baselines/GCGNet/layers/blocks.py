import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_len, output_len, d_model, d_ff, latent_dim=None):
        super(VAE, self).__init__()
        if latent_dim is None:
            latent_dim = d_model // 2
        self.encoder = nn.Sequential(
            nn.Linear(input_len, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, output_len)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # 从 N(0,1) 采样
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        z = self.reparameterize(mu, logvar)

        recon_x = self.decoder(z)

        return recon_x, mu, logvar


class mask_moe(nn.Module):
    def __init__(self, n_vars, patch_num, top_p=0.5, num_experts=3):
        super().__init__()
        self.num_experts = num_experts
        self.n_vars = n_vars
        self.patch_num = patch_num

        self.gate = nn.Linear(self.patch_num, num_experts, bias=False)
        self.noise = nn.Linear(self.patch_num, num_experts, bias=False)
        self.noisy_gating = True
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(2)
        self.top_p = top_p

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def cross_entropy(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return -torch.mul(x, torch.log(x + eps)).sum(dim=1).mean()

    def noisy_top_k_gating(self, x, is_training, noise_epsilon=1e-2):
        clean_logits = self.gate(x)
        if self.noisy_gating and is_training:
            raw_noise = self.noise(x)
            noise_stddev = ((self.softplus(raw_noise) + noise_epsilon))
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits

        # Convert logits to probabilities
        logits = self.softmax(logits)
        loss_dynamic = self.cross_entropy(logits)

        sorted_probs, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > self.top_p

        threshold_indices = mask.long().argmax(dim=-1)
        threshold_mask = torch.nn.functional.one_hot(threshold_indices, num_classes=sorted_indices.size(-1)).bool()
        mask = mask & ~threshold_mask

        top_p_mask = torch.zeros_like(mask)
        zero_indices = (mask == 0).nonzero(as_tuple=True)
        top_p_mask[
            zero_indices[0], zero_indices[1], sorted_indices[zero_indices[0], zero_indices[1], zero_indices[2]]] = 1

        sorted_probs = torch.where(mask, 0.0, sorted_probs)
        loss_importance = self.cv_squared(sorted_probs.sum(0))
        lambda_2 = 0.1
        loss = loss_importance + lambda_2 * loss_dynamic

        return top_p_mask, loss

    def forward(self, x, masks=None, is_training=None):
        # x [B, H, L, L]
        B, H, L, _ = x.shape
        device = x.device
        dtype = torch.float32

        mask_base = torch.eye(L, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        if self.top_p == 0.0:
            return mask_base, 0.0

        x = x.reshape(B * H, L, L)
        gates, loss = self.noisy_top_k_gating(x, is_training)
        gates = gates.reshape(B, H, L, -1).float()
        # [B, H, L, 3]

        if masks is None:
            masks = []
            N = L // self.n_vars
            for k in range(L):
                S = ((torch.arange(L) % N == k % N) & (torch.arange(L) != k)).to(dtype).to(device)
                T = ((torch.arange(L) >= k // N * N) & (torch.arange(L) < k // N * N + N)).to(dtype).to(device)
                ST = torch.ones(L).to(dtype).to(device) - S - T
                masks.append(torch.stack([S, T, ST], dim=0))
            # [L, 3, L]
            masks = torch.stack(masks, dim=0)

        mask = torch.einsum('bhli,lid->bhld', gates, masks) + mask_base

        return mask, loss


def mask_topk(x, alpha=0.5, largest=False):
    # B, L = x.shape[0], x.shape[-1]
    # x: [B, H, L, L]
    k = int(alpha * x.shape[-1])
    _, topk_indices = torch.topk(x, k, dim=-1, largest=largest)
    mask = torch.ones_like(x, dtype=torch.float32)
    mask.scatter_(-1, topk_indices, 0)  # 1 is topk
    return mask  # [B, H, L, L]


class Sparsifier(nn.Module):
    def __init__(self, n_vars, patch_num, top_p=0.5, num_experts=3):
        super().__init__()

        self.mask_moe = mask_moe(n_vars, patch_num, top_p, num_experts)

    def forward(self, adj, is_training, alpha=0.5, masks=None):
        adj = F.gelu(adj)
        adj = adj * mask_topk(adj, alpha)  # KNN
        mask, loss = self.mask_moe(adj, masks, is_training)
        adj = adj * mask
        return adj, loss
