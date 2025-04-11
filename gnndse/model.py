import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(128, 128)
        self.conv1 = GATConv(128, 64, edge_dim=128)
        self.conv2 = GATConv(128, 64, edge_dim=128)
        self.conv3 = GATConv(128, 64, edge_dim=128)
        self.conv4 = GATConv(128, 64, edge_dim=128)
        self.conv5 = GATConv(128, 64, edge_dim=128)
        self.conv6 = GATConv(128, 64, edge_dim=128)
        self.mlp1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
            )
        self.mlp2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
            )
        self.mlp3 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
            )
        self.mlp4 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
            )
        self.mlp5 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
            )
        self.mlp6 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
            )
        self.mlpg1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
            )
        self.mlpg2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
            )
        self.predict = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
            )
        
    def forward(self, data):
        x = self.emb(data.x)
        edge_index = data.edge_index
        edge_attr = self.emb(data.edge_attr)
        jkn = []
        jkn_ms = []
        x = self.conv1(x, edge_index, edge_attr)
        x = self.mlp1(x)
        jkn.append(x)
        jkn_ms.append(torch.sqrt(torch.sum(torch.square(x), dim=1)))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.mlp2(x)
        jkn.append(x)
        jkn_ms.append(torch.sqrt(torch.sum(torch.square(x), dim=1)))
        x = self.conv3(x, edge_index, edge_attr)
        x = self.mlp3(x)
        jkn.append(x)
        jkn_ms.append(torch.sqrt(torch.sum(torch.square(x), dim=1)))
        x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index, edge_attr)
        x = self.mlp4(x)
        jkn.append(x)
        jkn_ms.append(torch.sqrt(torch.sum(torch.square(x), dim=1)))
        x = self.conv5(x, edge_index, edge_attr)
        x = self.mlp5(x)
        jkn.append(x)
        jkn_ms.append(torch.sqrt(torch.sum(torch.square(x), dim=1)))
        x = F.dropout(x, training=self.training)
        x = self.conv6(x, edge_index, edge_attr)
        x = self.mlp6(x)
        jkn.append(x)
        jkn_ms.append(torch.sqrt(torch.sum(torch.square(x), dim=1)))
        jkn = torch.stack(jkn)
        jkn_ms = torch.stack(jkn_ms)
        ni = torch.max(jkn_ms, 0)[1]
        feature = jkn[ni, torch.arange(jkn.size(1))]
        hg = []
        shift = 0
        for i in range(data.num_graphs):
            fg = feature[shift : shift + len(data[i].x)]
            hg1 = self.mlpg1(fg)
            hg2 = self.mlpg2(fg)
            hg.append(torch.mean(torch.softmax(hg1, 0) * hg2, dim=0))
            shift += len(data[i].x)
        
        return self.predict(torch.stack(hg)).squeeze()
