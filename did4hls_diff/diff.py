import torch
import torch.nn as nn
import numpy as np
from attn import ResNet

class DM(nn.Module):
    def __init__(self, config_DM, len_prob):
        super().__init__()
        self.dim_feat = 256
        self.len_prob = len_prob
        self.res_model = ResNet(config_DM, self.dim_feat, len_prob)
        self.num_steps = config_DM['num_steps']
        self.beta = np.linspace(config_DM['beta_start'] ** 0.5, config_DM['beta_end'] ** 0.5, self.num_steps) ** 2
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().unsqueeze(1)
        
    def forward(self, x, prob):
        t = torch.randint(0, self.num_steps, [len(x)])
        current_alpha = self.alpha_torch[t].to(x.device)
        noise = torch.randn((len(x), self.dim_feat)).to(x.device)
        x = ((current_alpha ** 0.5) * x + ((1.0 - current_alpha) ** 0.5) * noise)
        predicted = self.res_model(x, t.to(x.device), prob)
        loss = ((noise - predicted) ** 2).sum() / len(predicted)
        
        return loss
        
    def generate(self, n_samples, prob):
        gen = []
        for i in range(n_samples):
            x_sample = torch.randn(self.dim_feat).to(prob.device)
            for t in range(self.num_steps - 1, -1, -1):
                predicted = self.res_model(x_sample, t, prob)
                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                x_sample = coeff1 * (x_sample - coeff2 * predicted)
                if t > 0:
                    noise = torch.randn(self.dim_feat).to(prob.device)
                    sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                    x_sample += sigma * noise
                    
            gen.append(x_sample)
            
        return torch.stack(gen)