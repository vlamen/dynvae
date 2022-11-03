import warnings
import torch 
from torch import nn 
import utils 
from torchtyping import TensorType
import torchsde 
import numpy as np

from .priors import EnergyNet, NoGradNet, NoGradNetFull, SumNetwork

""" 
This s a rather haphazard approach to put the code I'm using for the various experiments in one place.
Once the first few rounds of experiments are done, this should really be refactored into a module
"""
config = {}


def _add_to_config(obj, config=config):
    config[obj.__name__] = obj
    return obj


class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, online=False):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)  # batch_first just because I'm used to that
        # I might change the batch_first thing, but I already generated data with batch first
        self.lin = nn.Linear(hidden_size, output_size)
        self.online = online

    def forward(self, inp: TensorType['batch', 'time', 'input_size']) -> TensorType['batch', 'time', 'output_size']:
        if not self.online:
            inp = torch.flip(inp, dims=(1,))  # time reversal
        out, _ = self.gru(inp)
        out = self.lin(out)
        if not self.online:
            out = torch.flip(out, dims=(1,))  # reverse back to feed in right order
        return out


def safe_divide(numerator, denominator, eps=1e-7):
    safe_denominator = torch.where(torch.abs(denominator) > eps, denominator,
                                   torch.sign(denominator) * torch.full_like(denominator, eps))
    return numerator / safe_denominator


# this one comes from latent_sde_lorenz.py from the torchsde examples.
# it is used to train the SDE as a beta-vae with beta increasing from maxval/_iters to maxval
# so at the start you hardly use the kl-divergence as a loss term, and after _iters, you use it fully
class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self):
        return self._val


def train(sde, data, epochs, batch_size,
          shuffle=True,
          device=None,
          cache_size=10,
          kl_aneal_iters=1000,
          kl_max_val=1.,
          lr_init=1e-2,
          lr_gamma=.9997,
          report_every=20,
          on_epoch_end=None,
          noise_std=.1,
          test_data=None,
          print=print,
          t0=0.,
          t1=20.,
          nt=201,
          **kwargs):
    """

    :param sde: latent sde to be trained
    :param data: numpy array of data to train on
    :param epochs: number of epochs to train for
    :param batch_size: size of a batch
    :param shuffle: whether to shuffle the data every epoch
    :param device: device to use (default: cuda:0 if available else cpu)
    :param cache_size: number of batches to load onto the device at once  ---  this is mainly to minimize communication between cpu and gpu
    :param kl_aneal_iters: number of iterations  before the dkl is counted fully
    :param kl_max_val: beta from beta-vae
    :param lr_init:  initial learning rate
    :param lr_gamma: factor by which the learning rate decays after every batch
    :param report_every: how often to report (in terms of block, so cache_size*batch_size)
    :param on_epoch_end: optional callable to be called upon the end of any epoch (e.g. for flushing to an output file)
    :param noise_std: noise_std for sde forward pass
    :param test_data: data used for the validation loop
    :param print: print function to be used
    :param kwargs: any kwargs for the sde
    :return: model, losses

    losses is 
        a list of average epoch losses if test_data is None
        a list of (average epoch loss, test loss, test dkl, test reconstruction loss) if test_data is not None
    """
    import time
    if device is None:
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    sde.to(device)

    optimizer = torch.optim.Adam(sde.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_gamma)
    kl_scheduler = LinearScheduler(iters=kl_aneal_iters, maxval=kl_max_val)

    block_size = batch_size * cache_size
    number_of_blocks = data.shape[0] // block_size

    ts = torch.linspace(t0, t1, nt)
    losses = []

    for epoch in range(1, epochs+1):
        epoch_losses = []
        print(f'\nEpoch {epoch}')
        if shuffle:
            perm = np.random.permutation(data.shape[0])
            data = data[perm]

        t0 = time.time()
        for block_index in range(number_of_blocks):
            block = torch.from_numpy(data[block_index*block_size : (block_index+1)*block_size]).to(device=device, dtype=torch.float32)
            for batch in block.chunk(cache_size):
                optimizer.zero_grad()

                reconstruction_loss, dkl_loss = sde(batch, ts, noise_std=noise_std, **kwargs)
                loss = reconstruction_loss + kl_scheduler.val * dkl_loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                kl_scheduler.step()
                epoch_losses.append(loss.item())

            if (block_index+1) % report_every == 0:
                t1 = time.time()
                average_dt = (t1-t0) / (report_every*cache_size)
                lr_now = optimizer.param_groups[0]['lr']
                print(f'epoch {epoch} batch {(block_index+1)*cache_size}    {lr_now=}    took {average_dt} s / batch')
                print(f'{loss=:.4e},  {reconstruction_loss=:.4e},   {dkl_loss=:.4e}')
                t0 = time.time()
        average_epoch_loss = np.mean(epoch_losses)
        print(f'Training on epoch {epoch} finished.\n{average_epoch_loss=:.4e}')
        if test_data is not None:
            print('Starting validation loop')
            with torch.no_grad():
                num_test_blocks = test_data.shape[0] // block_size
                test_losses = []
                test_dkl = []
                test_rec = []
                for block_index in range(num_test_blocks):
                    block = torch.from_numpy(
                        test_data[block_index*block_size:(block_index+1)*block_size]
                    ).to(device=device, dtype=torch.float32)
                    for batch in block.chunk(cache_size):
                        rec_loss, dkl_loss = sde(batch, ts, noise_std=noise_std, **kwargs)
                        loss = rec_loss + kl_scheduler.val * dkl_loss
                        test_losses.append(loss.item())
                        test_dkl.append(dkl_loss.item())
                        test_rec.append(rec_loss.item())
            average_test_loss = np.mean(test_losses)
            average_test_dkl = np.mean(test_dkl)
            average_test_rec = np.mean(test_rec)
            print(f'Test loss: {average_test_loss:.4e}\nReconstruction: {average_test_rec:.4e}\nDkl: {average_test_dkl:.4e}\n')
            losses.append((average_epoch_loss, average_test_loss, average_test_dkl, average_test_rec))
        else:
            losses.append(average_epoch_loss)
        if on_epoch_end is not None:
            try:
                on_epoch_end()
            except Exception as e:
                warnings.warn(f'Tried to call on_epoch_end but got the following exception: {str(e)}')
    return losses


class FirstOrderLatentSDE(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, data_size, latent_size, context_size, hidden_size,
                 q_net,
                 p_net,
                 encoder,
                 model_difference=False,
                 take_gradient=False,
                 gamma=None,
                 beta=1/3,
                 learn_beta=False,
                 dt=1e-1):
        super().__init__()
        # Encoder.
        self.encoder = encoder  # Encoder(input_size=data_size, hidden_size=hidden_size, output_size=context_size)
        self.qz0_net = nn.Linear(context_size + data_size, latent_size + latent_size)

        # store the models
        self.f_net = q_net
        if not take_gradient:
            self.energy_landscape = None
            self._energy_gradient = None
            self.h_net = p_net
        else:
            self.energy_landscape = p_net
            self._energy_gradient = utils.GradNet(p_net)
            self.h_net = self._drift_from_energy_gradient

        # implement switch on model_difference
        self.model_difference = model_difference
        self.f_and_g = self.f_and_g_relative if self.model_difference else self.f_and_g_absolute

        # we use a constant diffusion
        if gamma is None:
            gamma = torch.ones((1, latent_size))
        else:
            gamma = utils.misc.new_tensor(gamma)
            if len(gamma.shape) == 1:
                gamma = torch.unsqueeze(gamma, 0)
            elif len(gamma.shape) > 2:
                raise(ValueError(f'Unexpected shape of gamma: {gamma.shape=}'))
        self.register_buffer('gamma', gamma)

        beta = utils.misc.new_tensor(beta)
        if learn_beta:
            self._beta = nn.Parameter(beta)
        else:
            self.register_buffer('_beta', beta)
        self._learn_beta = learn_beta

        self.register_buffer(  # only used if self._learn_beta
            '_diffusion',
            torch.sqrt(2/(self.beta*self.gamma))
        )

        self.projector = nn.Linear(latent_size, data_size)

        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))

        self._ctx = None

        self.dt = dt

    def contextualize(self, ctx):
        self._ctx = ctx  # A tuple of tensors of sizes (T,), (T, batch_size, d).

    def _drift_from_energy_gradient(self, z):
        return -self._energy_gradient(z)/self.gamma

    @property
    def diffusion(self):
        if self._learn_beta:
            return torch.sqrt(2/(self.beta * self.gamma))
        else:
            return self._diffusion

    @property
    def beta(self):
        if self._learn_beta:
            return nn.functional.softplus(self._beta)
        return self._beta

    def f_and_g_absolute(self, t, y):
        # separate y and dkl
        y, dkl = y[..., :-1], y[..., -1:]

        # get context
        ts, ctx = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)

        # get drift
        drift_y = self.f_net(torch.cat((y, ctx[:, i]), dim=1))
        prior_drift = self.h_net(y)

        # get diffusion
        diffusion_y = self.diffusion.repeat(y.shape[0], 1)

        # get the drift in dkl and put everything together
        drift_dkl = torch.square(safe_divide(drift_y-prior_drift, diffusion_y)).sum(-1, keepdim=True)
        drift = torch.cat([drift_y, drift_dkl], -1)
        diffusion = torch.cat([diffusion_y, torch.zeros_like(drift_dkl)], -1)

        return drift, diffusion

    def f_and_g_relative(self, t, y):
        # separate y and dkl
        y, dkl = y[..., :-1], y[..., -1:]

        # get context
        ts, ctx = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)

        # get drift
        prior_drift = self.h_net(y)
        difference = self.f_net(torch.cat((y, ctx[:, i]), dim=1))
        drift_y = prior_drift + difference

        # get diffusion
        diffusion_y = self.diffusion.repeat(y.shape[0], 1)

        # get the drift in dkl and put everything together
        drift_dkl = torch.square(safe_divide(difference, diffusion_y)).sum(-1, keepdim=True)
        drift = torch.cat([drift_y, drift_dkl], -1)
        diffusion = torch.cat([diffusion_y, torch.zeros_like(drift_dkl)], -1)

        return drift, diffusion

    def h(self, t, y):
        return self.h_net(y)

    def forward(self, xs: TensorType['batch', 'time', 'channels'], ts: TensorType['time'], noise_std, method="euler", dt=None, return_latent=False):
        if dt is None:
            dt = self.dt

        ctx = torch.cat([self.encoder(xs), xs], -1)  # (batch, time, context_size+data_size)
        self.contextualize((ts, ctx))

        qz0_mean, qz0_logstd = self.qz0_net(ctx[:, 0, :]).chunk(chunks=2, dim=-1)  # (batch, latent_size), (batch, latent_size)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)
        z0 = torch.cat([z0, torch.zeros_like(z0[..., :1])], -1)  # start dkl at 0, just add the initial value later

        zs = torchsde.sdeint(self, z0, ts, dt=dt, logqp=False, method=method)

        zs = zs.transpose(0, 1)  # batch, time, channels
        zs, dkls = zs[..., :-1], zs[..., -1]

        _xs = self.projector(zs)
        xs_dist = torch.distributions.Normal(loc=_xs, scale=noise_std)
        log_pxs = xs_dist.log_prob(xs).sum(dim=(1, 2)).mean(dim=0)  # sum over time and channels, mean over batch

        qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
        dkl0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0)
        dkl = dkls[..., -1].mean() + dkl0  # dkl from sde at final time + initial condition
        if not return_latent: 
            return -log_pxs, dkl # added a minus sign so both are just loss terms
        else:
            return -log_pxs, dkl, zs 

    @torch.no_grad()
    def sample(self, batch_size, ts, bm=None, dt=None):
        if dt is None:
            dt = self.dt
        eps = torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device)
        z0 = self.pz0_mean + self.pz0_logstd.exp() * eps
        zs = torchsde.sdeint(self, z0, ts, names={'drift': 'h', 'diffusion': '_g'}, dt=dt, bm=bm)
        # Most of the times in ML, we don't sample the observation noise for visualization purposes.
        _xs = self.projector(zs)
        return _xs

    @property
    def energy_gradient(self):
        # because I messed up
        return self._energy_gradient


class SecondOrderLatentSDE(nn.Module):
    """SecondOrderLatentSDE 
    the differential equation is a second order equation, split in location and momentum,
    with the diffusion only working on the momentum.
    """
    sde_type = 'ito'
    noise_type = 'diagonal'

    def __init__(self, data_size, latent_size, context_size, hidden_size,
        q_net,
        p_net, 
        encoder,
        model_difference=False,
        take_gradient=False,
        gamma=None,
        inverse_mass=None,
        beta=1/3,
        learn_beta=False,
        learn_gamma=False,
        learn_mass=False,
        dt=1e-1,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.context_size = context_size
        self.hidden_size = hidden_size

        # Encoder
        self.encoder = encoder  # Encoder(input_size=data_size, hidden_size=hidden_size, output_size=context_size)
        self.qz0_net = nn.Linear(context_size + data_size, 4*latent_size)

        self.control_network = q_net
        if not take_gradient:
            self.energy_landscape = None
            self.energy_gradient = None
            self.h_net = p_net
        else:
            self.energy_landscape = p_net
            self.energy_gradient = utils.GradNet(p_net)
            self.h_net = self._drift_from_energy_gradient
        
        # switch on model_difference
        self.model_difference = model_difference
        self.f_and_g = self.f_and_g_relative if self.model_difference else self.f_and_g_absolute

        # handle beta, gamma, and mass
        beta = utils.misc.new_tensor(beta)
        gamma = self._handle_input_tensor(gamma)
        inverse_mass = self._handle_input_tensor(inverse_mass)

        if learn_beta:
            self._beta = nn.Parameter(beta)
        else:
            self.register_buffer('_beta', beta)
        self._learn_beta = learn_beta

        if learn_gamma:
            self._gamma = nn.Parameter(gamma)
        else:
            self.register_buffer('_gamma', gamma)
        self._learn_gamma = learn_gamma
        
        if learn_mass:  # we mostly need the inverse of the mass matrix, so that's what we'll learn
            self._i_mass = nn.Parameter(inverse_mass)
        else:
            self.register_buffer('_i_mass', inverse_mass)
        self._learn_mass = learn_mass

        self._learn_diffusion = learn_beta or learn_gamma
        self.register_buffer('_fixed_diffusion', torch.sqrt(2*self._gamma/self._beta))
        self._soft_plus = nn.Softplus()

        # handle remaining business
        self.projector = nn.Linear(latent_size, data_size)

        self.pz0_mean = nn.Parameter(torch.zeros(1, 2*latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, 2*latent_size))

        self._ctx = None

        self.dt = dt

    def _handle_input_tensor(self, intensor):
        if intensor is not None:
            intensor = utils.misc.new_tensor(intensor)
            shape = intensor.shape
            if len(shape) == 1:
                intensor = torch.unsqueeze(intensor, 0)
            elif len(shape) > 2:
                raise ValueError(f'Unexpected shape: {intensor.shape}')
        else:
            intensor = torch.ones((1, self.latent_size))
        return intensor

    @property
    def beta(self):
        if self._learn_beta:
            return self._soft_plus(self._beta)
        return self._beta
    
    @property
    def gamma(self):
        if self._learn_gamma:
            return self._soft_plus(self._gamma)
        return self._gamma
    
    @property
    def inverse_mass(self):
        if self._learn_mass:
            return self._soft_plus(self._i_mass)
        return self._i_mass

    @property
    def mass(self):
        return 1/self.inverse_mass

    @property
    def diffusion(self):
        if self._learn_diffusion:
            return torch.sqrt(2*self.gamma/self.beta)
        return self._fixed_diffusion

    def _split_position_momentum(self, position_momentum):
        return torch.chunk(position_momentum, 2, -1)

    def _drift_from_energy_gradient(self, position_momentum):
        position, momentum = self._split_position_momentum(position_momentum)
        return - self.energy_gradient(position) - self.gamma * self.inverse_mass * momentum

    def contextualize(self, ctx):
        self._ctx = ctx

    def f_and_g_relative(self, t, z):
        position_momentum, dkl = z[..., :-1], z[..., -1:]

        # get the context
        ts, ctx = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts)-1)

        # get drift
        _, momentum = self._split_position_momentum(position_momentum)
        position_drift = self.inverse_mass * momentum
        prior_momentum_drift = self.h_net(position_momentum)
        difference = self.control_network(torch.cat([position_momentum, ctx[:, i]], -1))
        posterior_momentum_drift = prior_momentum_drift + difference

        # get diffusion
        momentum_diffusion = self.diffusion.repeat(z.shape[0], 1)
        position_diffusion = torch.zeros_like(momentum_diffusion)

        # get drift in dkl and put everything together
        dkl_drift = torch.square(safe_divide(difference, momentum_diffusion)).sum(-1, keepdim=True)
        drift = torch.cat([position_drift, posterior_momentum_drift, dkl_drift], -1)

        dkl_diffusion = torch.zeros_like(dkl_drift)
        diffusion = torch.cat([position_diffusion, momentum_diffusion, dkl_diffusion], -1)

        return drift, diffusion
    
    def f_and_g_absolute(self, t, z):
        position_momentum, dkl = z[..., :-1], z[..., -1:]

        # get the context
        ts, ctx = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts)-1)

        # get drift
        _, momentum = self._split_position_momentum(position_momentum)
        position_drift = self.inverse_mass * momentum
        prior_momentum_drift = self.h_net(position_momentum)
        posterior_momentum_drift = self.control_network(torch.cat([position_momentum, ctx[:, i]], -1))
        difference = prior_momentum_drift - posterior_momentum_drift

        # get diffusion
        momentum_diffusion = self.diffusion.repeat(z.shape[0], 1)
        position_diffusion = torch.zeros_like(momentum_diffusion)

        # get drift in dkl and put everything together
        dkl_drift = torch.square(safe_divide(difference, momentum_diffusion)).sum(-1, keepdim=True)
        drift = torch.cat([position_drift, posterior_momentum_drift, dkl_drift], -1)

        dkl_diffusion = torch.zeros_like(dkl_drift)
        diffusion = torch.cat([position_diffusion, momentum_diffusion, dkl_diffusion], -1)

        return drift, diffusion

    def forward(self,
                xs: TensorType['batch', 'time', 'channels'],
                ts: TensorType['time'],
                noise_std, method='euler',
                dt=None,
                return_latent=False):
        if dt is None:
            dt = self.dt

        ctx = torch.cat([self.encoder(xs), xs], -1)  # (batch, time, context_size + data_size)
        self.contextualize((ts, ctx))

        qz0_mean, qz0_logstd = self.qz0_net(ctx[:, 0, :]).chunk(2, -1)  # (batch, 2*latent_size), (batch, 2*latent_size)
        z0 = qz0_mean + qz0_logstd.exp()*torch.randn_like(qz0_mean)
        z0 = torch.cat([z0, torch.zeros_like(z0[..., :1])], -1)  # start dkl at 0, just add the initial value later

        # we don't use the adjoint method
        zs = torchsde.sdeint(self, z0, ts, dt=dt, logqp=False, method=method)
        zs = zs.transpose(0, 1)  # batch, time, channels
        position_momentums, dkls = zs[..., :-1], zs[..., -1]
        positions, momentums = self._split_position_momentum(position_momentums)

        _xs = self.projector(positions)
        xs_dist = torch.distributions.Normal(loc=_xs, scale=noise_std)
        log_pxs = xs_dist.log_prob(xs).sum(dim=(1, 2)).mean()  # sum over time and channels, mean over batch

        qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
        dkl0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0)
        dkl = dkls[..., -1].mean() + dkl0  # dkl from sde at final time + initial condition

        if not return_latent:
            return -log_pxs, dkl
        else:
            return -log_pxs, dkl, positions, momentums

    def h(self, t, position_momentum):
        position, momentum = self._split_position_momentum(position_momentum)

        position_drift = self.inverse_mass * momentum
        momentum_drift = self.h_net(position_momentum)
        drift = torch.cat([position_drift, momentum_drift], -1)

        return drift

    def _g(self, t, position_momentum):
        momentum_diffusion = self.diffusion.repeat(position_momentum.shape[0], 1)
        position_diffusion = torch.zeros_like(momentum_diffusion)
        diffusion = torch.cat([position_diffusion, momentum_diffusion])
        return diffusion

    @torch.no_grad()
    def sample(self, batch_size, ts, bm=None, dt=None, return_latent=False):
        if dt is None:
            dt = self.dt
        eps = torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device)
        z0 = self.pz0_mean + self.pz0_logstd.exp()*eps 
        zs = torchsde.sdeint(self, z0, ts, names={'drift': 'h', 'diffusion': '_g'}, dt=dt, bm=bm)

        positions, momentums = self._split_position_momentum(zs)
        _xs = self.projector(positions)
        if return_latent:
            return _xs, positions, momentums  
        return _xs


class PTanh(nn.Module):
    """
    Used as an activation function to ensure that out put is within fixed bounds
    """

    def __init__(self, p=1., learn_p=True):
        super().__init__()

        if learn_p:
            self.p = nn.Parameter(utils.misc.new_tensor(p))
        else:
            self.register_buffer('p', utils.misc.new_tensor(p))
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.p*self.tanh(x/self.p)


def get_control_network(data_size, latent_size, context_size, hidden_size, overdamped=True, use_tanh=False):
    return nn.Sequential(
                        nn.Linear((1 if overdamped else 2)*latent_size + context_size + data_size, hidden_size),
                        nn.Softplus(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.Softplus(),
                        nn.Linear(hidden_size, latent_size),
                        PTanh(learn_p=True) if use_tanh else nn.Identity()
                        )


@_add_to_config
def get_h_net_grad(latent_size):
    """ 
    Just gives you EnergyNet(latent_size)
    """
    return EnergyNet(latent_size)


@_add_to_config
def get_h_net_full(latent_size):
    """ 
    Just gives you NoGradNetFull(latent_size)
    """
    return NoGradNetFull(latent_size)


@_add_to_config
def get_h_net_split(latent_size):
    """ 
    gives you a SumNetwork with both components being instances of NoGradNet(latent_size)
    """
    return SumNetwork(
        NoGradNet(latent_size),
        NoGradNet(latent_size)  # this might be overkill for the momentum network, but whatever
    )


@_add_to_config
def get_h_net_no_grad(latent_size):
    """
    just gives you NoGradNet(latent_size)
    """
    return NoGradNet(latent_size)


def create_model(profile, settings):
    """create_model create a model from a profile and settings

    :param profile: dict-like containing a function at key 'get_h_net', and two bools at keys 'take_gradient' and 'model_difference'
    :param settings: a utils.settings.Settings-like object containing all relevant parameters

    settings should include:
    * data_size: int
    * latent_size: int
    * hidden_size: int
    * context_size: int
    * online: whether to make a first order SDE model (True) or a second order SDE model (False)
    * overdamped: bool (whether)
    """
    overdamped = settings.overdamped
    LatentSDE = FirstOrderLatentSDE if overdamped else SecondOrderLatentSDE

    h_net = profile['get_h_net'](settings.latent_size)
    if 'h_net_init' in profile:
        h_net.apply(profile['h_net_init'])
    elif 'h_net_init' in settings:
        h_net.apply(settings.h_net_init)

    control_net = get_control_network(
        settings.data_size,
        settings.latent_size,
        settings.context_size,
        settings.hidden_size,
        overdamped=overdamped,
        use_tanh=settings.get('controll_use_tanh', False)
    )
    if 'control_init' in settings:  # modify the initialization weights of control_net using a function
        control_net.apply(settings.control_init)

    encoder = GRUEncoder(settings.data_size, settings.hidden_size, settings.context_size, online=settings.online)
    if 'encoder_init' in settings:
        encoder.apply(settings.encoder_init)

    relevant_keys = ['beta', 'learn_beta', 'dt']
    if not overdamped:
        relevant_keys += ['inverse_mass', 'learn_mass', 'gamma', 'learn_gamma']

    kwargs = {
        key: settings[key] 
        for key in relevant_keys
        if key in settings 
    }

    return LatentSDE(
        settings.data_size,
        settings.latent_size, 
        settings.context_size,
        settings.hidden_size,
        control_net,
        h_net,
        encoder,
        model_difference=profile['model_difference'],
        take_gradient=profile['take_gradient'], 
        **kwargs
    )


def run_experiment(
        profile,
        settings,
        train_data,
        test_data=None,
        plot_losses=False,
        on_epoch_end=None,
        plot_start=2,
        print=print):
    """run_experiment run one experiment

    :param profile: dict-like containing a function at key 'get_h_net', and two bools at keys 'take_gradient' and 'model_difference'
    :param settings: utils.settings.Settings-like object containing relevant settings
    :param train_data: data to be used for training
    :param test_data: data to be used for the test loop
    :param plot_losses: whether to attempt to plot the loss curve after training
    :param on_epoch_end: optional callable to be passed to train
    :param plot_start: what epoch to start plotting at (index of losses to start the plot)
    :param print: print function to be used

    :return: trained_model, losses (list)

    relevant settings include:
    * data_size: int
    * latent_size: int
    * hidden_size: int
    * context_size: int
    * online: bool 
    * overdamped: bool (whether to make a first order SDE model (True) or a second order SDE model (False))
    
    * max_attempts: int  how often to attempt to run the experiment before giving up on nans
    * epochs: int
    * batch_size: int
    * gpu: int
    * noise_std: float
    """
    device = torch.device(f'cuda:{settings.gpu}') if torch.cuda.is_available() else torch.device('cpu')
    cpu = torch.device('cpu')
    train_kwargs = [
        'cache_size',
        'kl_aneal_iters',
        'kl_max_val',
        'lr_init',
        'lr_gamma',
        'report_every',
        'noise_std',
        'method',
        'dt',
        'adjoint'
    ]
    train_kwargs = {
        key: settings[key]
        for key in train_kwargs
        if key in settings
    }

    if 't0' in settings:
        t0 = settings.t0
    else:
        t0 = 0.
    if 't1' in settings:
        t1 = settings.t1
    else:
        t1 = train_data.shape[1]//10
    if 'nt' in settings:
        nt = settings.nt
    else:
        nt = train_data.shape[1]

    for attempt in range(settings.max_attempts):
        print(60*'=')
        print(f'Attempt {attempt}')
        try:
            model = create_model(profile, settings)
            losses = train(
                model,
                train_data,
                epochs=settings.epochs,
                batch_size=settings.batch_size,
                shuffle=True,
                device=device,
                test_data=test_data,
                print=print,
                on_epoch_end=on_epoch_end,
                t0=t0,
                t1=t1,
                nt=nt,
                **train_kwargs
            )
            model.to(cpu)
            if plot_losses:
                try:
                    import matplotlib.pyplot as plt
                    plt.plot(np.arange(plot_start, len(losses)), losses[plot_start:])
                    plt.show()
                except Exception as e:
                    print('Failed to plot loss')
                    print('Received the following exception:')
                    print(repr(e))
            break 
        except ValueError as e:
            print('Got ValueError:\n', str(e))
    else:
        return None, None  # no successfully trained model
    return model, losses


# some functions for experimenting with weight initialization
@_add_to_config
def zero_init(module):
    """zero_init 
    meant to initialize the weights and biases of linear layers as zeroes

    :param module: any torch.nn.Module

    recommended use: module.apply(zero_init)
    """
    if isinstance(module, nn.Linear):
        module.weight.data.fill_(0.)
        if module.bias is not None:
            module.bias.data.fill_(0.)


@_add_to_config
def fractional_init(fraction):
    """fractional_init 
    higher order function for creating an initialization function that multiplies the original weights and biases with fraction

    :param fraction: scalar

    how to use with settings:
    in settings file write e.g. "config['fractional_init'](.1)"
    """
    def _fractional_init(module):
        f"""_fractional_init
        meant to adjust the weights and biases of linear layers by multiplying by {fraction}

        :param module: any torch.nn.Module

        recommended use: module.apply(_fractional_init)
        """
        if isinstance(module, nn.Linear):
            with torch.no_grad():
                module.weight *= fraction 
                if module.bias is not None:
                    module.bias *= fraction 
    _fractional_init.__name__ = 'fractional_init'
    _fractional_init._arguments = (fraction, )
    return _fractional_init
