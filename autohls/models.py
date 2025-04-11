import torch.nn as nn

class Predictor(nn.Module):
    def __init__(self, n_type):
        super().__init__()
        self.dnn = nn.Sequential(
            nn.BatchNorm1d(n_type),
            nn.Linear(n_type, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p = 0.2),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.ReLU()
        )
        
    def forward(self, x):
        
        return self.dnn(x)
    
class Scorer(nn.Module):
    def __init__(self, n_type):
        super().__init__()
        self.dnn = nn.Sequential(
            nn.BatchNorm1d(n_type),
            nn.Linear(n_type, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p = 0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        
        return self.dnn(x).squeeze()