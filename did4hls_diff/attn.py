import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_torch_trans(config_DM):
    encoder_layer = nn.TransformerEncoderLayer(d_model=config_DM["d_model"], nhead=config_DM["n_heads"], dim_feedforward=64, activation="gelu", batch_first=True)
    
    return nn.TransformerEncoder(encoder_layer, num_layers=config_DM["n_enc_layers"])

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim):
        super().__init__()
        self.register_buffer('diffusion_embedding', self._build_embedding(num_steps, embedding_dim / 2), persistent=False)
        self.projection1 = nn.Linear(embedding_dim, embedding_dim)
        self.projection2 = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, diffusion_step):
        x = self.diffusion_embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        
        return x
    
    def _build_embedding(self, num_steps, dim):
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = (10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0))
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        
        return table

class ResNet(nn.Module):
    def __init__(self, config_DM, dim_feat, len_prob):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Linear(dim_feat, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, dim_feat),
        )
        self.residual_layers = nn.ModuleList([Triplet_cor(config_DM, dim_feat, len_prob) for _ in range(config_DM['n_res_layers'])])
            
    def forward(self, x, diffusion_step, prob):
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step, prob)
            skip.append(skip_connection)
            
        output = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        
        return self.dec(output)

class Triplet_cor(nn.Module):
    def __init__(self, config_DM, dim_feat, len_prob):
        super().__init__()
        self.diffusion_embedding = DiffusionEmbedding(config_DM['num_steps'], config_DM['diffusion_embedding_dim'])
        self.diffusion_projection = nn.Linear(config_DM['diffusion_embedding_dim'], dim_feat)
        self.proj_prob = nn.Sequential(
            nn.Linear(len_prob, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * dim_feat),
        )
        self.proj_enc = nn.Sequential(
            nn.Linear(dim_feat, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * dim_feat),
        )
        self.proj_split = nn.Sequential(
            nn.Linear(dim_feat, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * dim_feat),
        )
        
    def forward(self, x, diffusion_step, prob):
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        diffusion_emb = self.diffusion_projection(diffusion_emb)
        y = x + diffusion_emb
        y = self.proj_enc(y) + self.proj_prob(prob)
        gt, ft = torch.chunk(y, 2, dim=-1)
        y = torch.sigmoid(gt) * torch.tanh(ft)
        y = self.proj_split(y)
        residual, skip = torch.chunk(y, 2, dim=-1)
        
        return (x + residual) / torch.sqrt(torch.tensor(2)), skip