import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as TD


class Base(nn.Module):
    def __init__(self):
        super().__init__()

    def _mlp(self, n_in, n_out):
        return nn.Sequential(
            nn.Linear(n_in, n_out),
            nn.BatchNorm1d(n_out),
            nn.ReLU()
        )

class Encoder(Base):
    def __init__(self, input_size, n_latent):
        super().__init__()

        n = []
        n.append(1024)
        n.append(512)
        n.append(256)
        
        self.encoder = nn.Sequential(
            self._mlp(input_size, n[0]),
            self._mlp(n[0], n[1]),
            self._mlp(n[1], n[2]),
        )    
        
        self.mu = nn.Linear(n[-1], n_latent)
        self.sigma = nn.Linear(n[-1], n_latent)

    def forward(self, x):
        encoder_output = self.encoder(x)
        mu = self.mu(encoder_output)
        log_std = self.sigma(encoder_output)
        return mu, log_std
        

class Decoder(Base):
    def __init__(self, input_size, n_latent):
        super().__init__()

        output_shape = input_size

        n = []
        n.append(256)
        n.append(512)
        n.append(1024)

        self.decoder = nn.Sequential(
            self._mlp(n_latent, n[0]),
            self._mlp(n[0], n[1]),
            self._mlp(n[1], n[2]),
            nn.Linear(n[-1], output_shape) 
        )    

    def forward(self, z):
        out = self.decoder(z)
        return out


class VAE(nn.Module):
    def __init__(self, input_size, n_latent, beta, device):
        super().__init__()

        self.device = device
        
        self.input_size = input_size
        self.n_latent = n_latent
        self.beta = beta

        self.encoder = Encoder(input_size, n_latent)
        self.decoder = Decoder(input_size, n_latent)

    def prior(self, n):
        d = TD.Normal(0, 1)
        z = d.sample(torch.tensor((n, self.n_latent)))
        z = z.to(self.device)   
        return z

    def forward(self, x):
        mu_z, log_std_z = self.encoder(x)
        
        eps = self.prior(x.shape[0])
        z = mu_z + eps*(torch.exp(log_std_z))
        x_recon = self.decoder(z)
        return mu_z, log_std_z, x_recon
        
    def loss(self, x):
        mu_z, log_std_z, x_recon = self(x)
        recon_loss = F.mse_loss(x_recon, x, reduce=False)
        recon_loss_mean = recon_loss.mean() 
        kl_loss = -0.5 * torch.mean(1 + log_std_z - mu_z ** 2 - log_std_z.exp())

        return {
            'elbo_loss': recon_loss_mean + self.beta * kl_loss, 
            'recon_loss': recon_loss_mean,
            'kl_loss': kl_loss,
            'recon_quality': recon_loss
        }

    def sample(self, n):
        with torch.no_grad():
            z = self.prior(n)
            x_recon = self.decoder(z)                   
            samples = torch.clamp(x_recon, -1, 1)
        return samples.cpu().numpy() * 0.5 + 0.5

    
class ad_Model(nn.Module):
    def __init__(self, dgm_model, inputs, n_latent, beta, device):
        super().__init__()

        X, _ = inputs
        
        # dgm_model
        input_size = X.shape[1]
        self.vae = dgm_model(input_size, n_latent, beta, device)
        self.device = device
        
    def forward(self, inputs):
        X, _ = inputs
        X = X.to(torch.float32).to(self.device)
        return X
    
    def loss(self, X):
        X = self(X)
        return self.vae.loss(X)
