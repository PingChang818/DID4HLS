import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class GAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_t = nn.Embedding(60, 64)
        self.emb_b = nn.Linear(1, 64)
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
        x = self.emb_t(data.x[:, 0]) + self.emb_b(data.x[:, 1].to(torch.float).unsqueeze(1))
        edge_index = data.edge_index
        edge_attr = self.emb_t(data.edge_attr)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.conv3(x, edge_index, edge_attr)
        attn = []
        for i in range(len(data.ptr) - 1):
            attn.append(torch.mean(x[data.ptr[i] : data.ptr[i + 1]], 0))
            
        return torch.stack(attn)
    
class Generator(nn.Module):
    def __init__(self, len_prob):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(len_prob, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )
        self.mlp = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
    
    def forward(self, prob):
        z = torch.randn(len(prob), 64).to(prob.device)
    
        return self.mlp(torch.cat([z, self.proj(prob)], dim = -1))

class Discriminator(nn.Module):
    def __init__(self, len_prob):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(len_prob, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, prob):
        x = torch.cat([x, self.proj(prob)], dim = -1)
        
        return self.mlp(x)