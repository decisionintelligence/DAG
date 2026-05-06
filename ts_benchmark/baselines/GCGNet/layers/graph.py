import torch
import torch.nn as nn
import torch.nn.functional as F

from ts_benchmark.baselines.GCGNet.layers.blocks import VAE


class GraphLearner(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.proj_1 = nn.Linear(d_model, d_model)
        self.proj_2 = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, P_NUM, _ = x.shape
        proj_1_x = self.proj_1(x).view(B, P_NUM, self.n_heads, -1).permute(0, 2, 1, 3)
        proj_2_x = self.proj_2(x).view(B, P_NUM, self.n_heads, -1).permute(0, 2, 1, 3)
        adj = F.gelu(torch.einsum('bhid,bhjd->bhij', proj_1_x, proj_2_x)).contiguous()
        adj = 0.5 * (adj + adj.transpose(-1, -2))
        return adj


class GCN(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, adj, x):
        # adj [B, H, L, L]
        B, L, D = x.shape
        x = self.proj(x).view(B, L, self.n_heads, -1)  # [B, L, H, D_]
        adj = F.normalize(adj, p=1, dim=-1)
        x = torch.einsum("bhij,bjhd->bihd", adj, x).contiguous()  # [B, L, H, D_]
        x = x.view(B, L, -1)
        return x


# class GraphVAE(nn.Module):
#     def __init__(self, d_model, d_ff, n_heads):
#         super(GraphVAE, self).__init__()
#
#         self.graph_learner = GraphLearner(d_model, n_heads)
#         self.vae = VAE(d_model, d_model, d_model, d_ff)
#
#     def forward(self, nodes):
#         adj = self.graph_learner(nodes)
#         nodes_z, mu, logvar = self.vae(nodes)
#         return adj, mu, logvar

class GraphEmbedding(nn.Module):
    def __init__(self, nodes_num, d_model, rank):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(nodes_num, d_model),
            nn.ReLU(),
            nn.Linear(d_model, rank)
        )

    def forward(self, adj):
        # adj: [batch, N, N]
        U = self.proj(adj)
        return U


class GraphVAE(nn.Module):
    def __init__(self, d_model, d_ff, max_num_nodes):
        super(GraphVAE, self).__init__()

        output_dim = max_num_nodes * (max_num_nodes + 1) // 2
        self.vae = VAE(max_num_nodes * max_num_nodes, output_dim, d_model, d_ff)

        self.max_num_nodes = max_num_nodes

    def recover_adj_lower(self, batch_out):
        batch_size = batch_out.size(0)
        device = batch_out.device
        adj_batch = torch.zeros(batch_size, self.max_num_nodes, self.max_num_nodes, device=device)
        triu_indices = torch.triu_indices(self.max_num_nodes, self.max_num_nodes)
        adj_batch[:, triu_indices[0], triu_indices[1]] = batch_out

        return adj_batch

    def recover_full_adj_from_lower(self, lower_batch):
        diag = torch.diagonal(lower_batch, dim1=1, dim2=2)  # [batch_size, N]
        diag_matrix = torch.zeros_like(lower_batch)
        diag_matrix[:, torch.arange(self.max_num_nodes), torch.arange(self.max_num_nodes)] = diag
        lower_transpose = lower_batch.transpose(1, 2)
        full_adj = lower_batch + lower_transpose - diag_matrix
        return full_adj

    def forward(self, input_features):
        graph_h = input_features.view(-1, self.max_num_nodes * self.max_num_nodes)
        out, mu, logvar = self.vae(graph_h)
        recon_adj_lower = self.recover_adj_lower(out)
        recon_adj_tensor = self.recover_full_adj_from_lower(recon_adj_lower)
        return recon_adj_tensor, mu, logvar


class GraphDiscriminator(nn.Module):
    def __init__(self, nodes_num, d_model, d_ff, n_heads, rank):
        super(GraphDiscriminator, self).__init__()
        self.nodes_num = nodes_num
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.rank = rank

        self.graph_learner = GraphLearner(self.d_model, self.n_heads)
        self.graph_vae = GraphVAE(self.d_model, self.d_ff, self.nodes_num)

    def forward(self, x):
        # x: [B, Patch num, d_model]
        adj = self.graph_learner(x)
        adj = adj.view(-1, self.nodes_num, self.nodes_num)
        adj, mu, logvar = self.graph_vae(adj)
        adj = adj.view(-1, self.n_heads, self.nodes_num, self.nodes_num)
        return adj, mu, logvar


class GCNStack(nn.Module):
    def __init__(self, d_model, n_heads, e_layers=2, dropout=0):
        super().__init__()
        self.layers = nn.ModuleList([GCN(d_model, n_heads) for _ in range(e_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, adj, x):
        for i, layer in enumerate(self.layers):
            x = layer(adj, x)
            if i < len(self.layers) - 1:  # 最后一层不激活不dropout
                x = F.relu(x)
                x = self.dropout(x)
        return x
