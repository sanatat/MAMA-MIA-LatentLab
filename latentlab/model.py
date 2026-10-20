"""Unsupervised VAE representation and out-of-cohort research surrogate."""
import random
import numpy as np
import torch
from torch import nn

class DenseVAE(nn.Module):
    def __init__(self, n_features, latent=4, hidden=24):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, n_features))

    def forward(self, x):
        enc = self.encoder(x)
        mu, logvar = self.mu(enc), self.logvar(enc).clamp(-12, 12)
        if self.training:
            z = mu + torch.exp(.5*logvar)*torch.randn_like(mu)
        else:
            z = mu
        return self.decoder(z), mu, logvar


def train_vae(train_x, latent=4, epochs=160, seed=42, beta=0.02):
    if len(train_x) < 10 or train_x.shape[1] < 2:
        raise ValueError("At least 10 training patients and two features are required")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    net = DenseVAE(train_x.shape[1], latent=min(latent,train_x.shape[1]))
    optim = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    x = torch.tensor(np.asarray(train_x), dtype=torch.float32)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x),
                 batch_size=min(64,len(x)), shuffle=True,
                 generator=torch.Generator().manual_seed(seed))
    trace=[]
    for _ in range(epochs):
        net.train(); losses=[]
        for (batch,) in loader:
            optim.zero_grad()
            recon, mu, logvar = net(batch)
            mse = torch.mean((recon-batch)**2)
            kl = (-0.5*(1+logvar-mu.square()-logvar.exp())).sum(dim=1).mean()
            loss = mse + beta*kl
            loss.backward(); optim.step(); losses.append(float(loss.detach()))
        trace.append(float(np.mean(losses)))
    net.eval()
    return net, trace

@torch.no_grad()
def project(net, x):
    _, mu, _ = net(torch.tensor(np.asarray(x), dtype=torch.float32))
    return mu.numpy()
