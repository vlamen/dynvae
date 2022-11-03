import torch 
from torch import nn

class EnergyNet(nn.Module):
    def __init__(self, latent_dim, high_size=96, low_size=16, res_blocks=4):
        super().__init__()
        self.in_block = nn.Sequential(
            nn.Linear(latent_dim, high_size),
            nn.CELU(),
            nn.Linear(high_size, low_size)
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(low_size + latent_dim, high_size),
                nn.Tanh(),
                nn.Linear(high_size, low_size)
            )
            for _ in range(res_blocks)
        ])
        self.out_block = nn.Sequential(
            nn.Linear(low_size, high_size),
            nn.CELU(),
            nn.Linear(high_size, 1)
        )

        self._quadratic_term = torch.nn.Parameter(torch.normal(torch.zeros(())))

    @property
    def quadratic_term(self):
        return nn.functional.softplus(self._quadratic_term)

    def forward(self, x):
        x1 = self.in_block(x)
        for block in self.blocks:
            x1 += block(torch.cat([x1, x], dim=-1))
        return self.out_block(x1) + self.quadratic_term*torch.sum(torch.square(x), dim=-1, keepdim=True)


class NoGradNet(nn.Module):
    """
    latent_dim -> latent_dim
    high capacity
    """
    def __init__(self, latent_dim, high_size=96, low_size=16, res_blocks=4):
        super().__init__()
        self.in_block = nn.Sequential(
            nn.Linear(latent_dim, high_size),
            nn.CELU(),
            nn.Linear(high_size, low_size)
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(low_size + latent_dim, high_size),
                nn.Tanh(),
                nn.Linear(high_size, low_size)
            )
            for _ in range(res_blocks)
        ])
        self.out_block = nn.Sequential(
            nn.Linear(low_size, high_size),
            nn.CELU(),
            nn.Linear(high_size, latent_dim)
        )

        self._quadratic_term = torch.nn.Parameter(torch.normal(torch.zeros(())))

    def forward(self, x):
        x1 = self.in_block(x)
        for block in self.blocks:
            x1 += block(torch.cat([x1, x], dim=-1))
        return self.out_block(x1) - 2*torch.nn.functional.softplus(self._quadratic_term)*x
    

class NoGradNetFull(nn.Module):
    """
    2*latent_dim -> latent_dim
    """
    def __init__(self, latent_dim, high_size=96, low_size=16, res_blocks=4):
        super().__init__()
        self.in_block = nn.Sequential(
            nn.Linear(2*latent_dim, high_size),
            nn.CELU(),
            nn.Linear(high_size, low_size)
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(low_size + 2*latent_dim, high_size),
                nn.Tanh(),
                nn.Linear(high_size, low_size)
            )
            for _ in range(res_blocks)
        ])
        self.out_block = nn.Sequential(
            nn.Linear(low_size, high_size),
            nn.CELU(),
            nn.Linear(high_size, latent_dim)
        )

        self._quadratic_term = torch.nn.Parameter(torch.normal(torch.zeros(())))

    def forward(self, x):
        x1 = self.in_block(x)
        for block in self.blocks:
            x1 += block(torch.cat([x1, x], dim=-1))
            
        quadratic_term_components = 2*torch.nn.functional.softplus(self._quadratic_term)*x
        component_1, component_2 = quadratic_term_components.chunk(2,-1)
        quadratic_term = component_1 + component_2
        return self.out_block(x1) - quadratic_term


class SumNetwork(nn.Module):
    def __init__(self, position_net, momentum_net):
        super().__init__()
        self.position_net = position_net
        self.momentum_net = momentum_net

    def forward(self, x):
        position, momentum = torch.chunk(x, 2, dim=-1)
        return self.position_net(position) + self.momentum_net(momentum)
