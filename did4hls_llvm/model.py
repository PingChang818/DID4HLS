import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class GAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(128, 64)
        self.conv1 = GATv2Conv(64, 128, edge_dim=64)
        self.conv2 = GATv2Conv(128, 128, edge_dim=64)
        self.conv3 = GATv2Conv(128, 256, edge_dim=64)
        self.predict = nn.Sequential(
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 1)
            )
        
    def forward(self, data):
        x = self.emb(data.x)
        edge_index = data.edge_index
        edge_attr = self.emb(data.edge_attr)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.conv3(x, edge_index, edge_attr)
        attn = []
        for i in range(len(data.ptr) - 1):
            attn.append(torch.mean(x[data.ptr[i] : data.ptr[i + 1]], 0))
            
        return torch.stack(attn)
    
class CVAE(nn.Module):
    def __init__(self, len_prob):
        super().__init__()
        self.len_prob = len_prob
        self.encoder = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ELU(),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 512),
            nn.ELU(),
            nn.Linear(512, 1024),
            nn.ELU(),
            nn.Linear(1024, 1024),
            nn.ELU(),
            nn.Linear(1024, 256)
        )
        self.mean_proj = nn.Linear(64, 64)
        self.log_var_proj = nn.Linear(64, 64)
        self.proj = nn.Sequential(
            nn.Linear(len_prob, 128),
            nn.ELU(),
            nn.Linear(128, 64)
        )
        
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        
        return mean + eps * std
    
    def forward(self, x, prob):
        enc = self.encoder(x)
        mean = self.mean_proj(enc)
        log_var = self.log_var_proj(enc)
        z = self.reparameterize(mean, log_var)
        dec = self.decoder(z + self.proj(prob))
        
        return dec, mean, log_var
