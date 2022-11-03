import torch
from torch import nn 
import numpy as np
from utils.gradients import GradNet
from utils.misc import new_tensor
from typing import Union
import warnings
import abc


""" 
This file contains functions and classes to help investigate energy landscapes.
"""


class _BaseSampler(nn.Module):
    def __init__(self, energy_function):
        super().__init__()
        self._energy_function = energy_function
        self._energy_function_grad = GradNet(energy_function)

    def energy_function(self, q):
        """
        The energy function based on the energy perturbation
        :param q: position variable
            shape:(*batch_dims, latent_dim)
        :return: the potential energy at q
            shape: (*batch_dims, 1)
        """
        return self._energy_function(q)

    def nabla_energy(self, q):
        """
        returns the gradient of the energy field at q
        :param q: location at which to calculate the gradient of the energy
        :return: tensor of the same shape as q
        """
        return self._energy_function_grad(q)


class GHMCSampler(_BaseSampler):
    """
    Generalized Hybrid Monte-Carlo sampler [Horowitz (1991)]
    See Algorithm 2.11 in 'Free energy computations - a mathematical perspective' by Lelievre, Rousset, and Stoltz
    """

    def __init__(self,
                 dim,
                 energy_function,
                 inverse_temperature,
                 inverse_mass=None,
                 friction=None,
                 initial_distribution=None,
                 step_size=None,
                 strict=True,
                 steps=500,
                 track_acceptance_rate=False
                 ):
        """
        initialization
        :param dim: dimensionality of samples
        :param energy_function: nn.Module for computing the energy function
        :param inverse_temperature: beta (float)
        :param inverse_mass: diagonal of inverse mass matrix
        :param friction: diagonal of friction matrix
        :param initial_distribution: InitialDistribution object
            NB should sample for both location and momentum
            i.e. should return a [batch_size, 2*dim] tensor
        :param step_size: step size to be used
            defaults to 1/16 times the computed maximal step size
        :param track_acceptance_rate: if True, keeps track of acceptance rate
        :param strict: if True, raises errors instead of warnings
        """
        super().__init__(energy_function)
        self.dim = dim
        self.register_buffer(
            'inverse_temperature',
            new_tensor(inverse_temperature)
        )
        self.register_buffer(
            'friction',
            new_tensor(friction).reshape(1, dim) if friction is not None else torch.ones((1, dim))
        )
        self.register_buffer(
            'inverse_mass',
            new_tensor(inverse_mass).reshape(1, dim) if inverse_mass is not None else torch.ones((1, dim))
        )

        if initial_distribution is None:
            self.initial_distribution = DiagonalGaussianDistribution(2*dim)
        else:
            self.initial_distribution = initial_distribution

        self.strict = strict

        # see equation 2.48 in 'Free energy computations'
        max_step_size = torch.max(2./(self.inverse_mass * self.friction))
        if step_size is not None:
            fl_step_size = step_size.item() if isinstance(step_size, torch.Tensor) else step_size
            if fl_step_size > max_step_size.item():
                if self.strict:
                    raise ValueError(f'Step size {fl_step_size} is larger than maximum step size {max_step_size.item()}')
                else:
                    warnings.warn(f'Step size {fl_step_size} is larger than maximum step size {max_step_size.item()}')
        else:
            step_size = max_step_size/16
        step_size = new_tensor(step_size)
        self.register_buffer('step_size', step_size)
        self.register_buffer('half_step_size', step_size/2)
        self.register_buffer('quarter_step_size', step_size/4)

        # default matrices for mid-point euler
        # note that we use half step sizes for the mid-point euler steps
        self.register_buffer(
            '_a_default',
            self._a(self.half_step_size)
        )
        self.register_buffer(
            '_b_default',
            self._b(self.half_step_size)
        )

        self.steps = steps

        self.track_acceptance_rate = track_acceptance_rate
        self._acceptance_rates = []

    @property
    def acceptance_rate(self):
        return sum(self._acceptance_rates) / max(1, len(self._acceptance_rates))

    def hamiltonian(self, q, p):
        """
        The full hamiltonian in q and p
        :param q: location
        :param p: momentum
        :return: H(q,p)
        """
        return self.energy_function(q) + torch.sum(p * self.inverse_mass * p,
                                                   -1,
                                                   keepdim=True) / 2.

    def _a(self, step_size=None):
        """
        a matrix for midpoint euler step
        :param step_size:
        :return:
        """
        if step_size is None:
            return self._a_default
        # return (1. - step_size / 2 * self.friction * self.inverse_mass) / \
        #        (1. + step_size / 2 * self.friction * self.inverse_mass)
        return (1. - step_size / 2 * self.friction) / \
               (1. + step_size / 2 * self.friction)

    def _b(self, step_size=None):
        """
        b matrix for midpoint euler step
        :param step_size:
        :return:
        """
        if step_size is None:
            return self._b_default
        # return torch.sqrt(step_size * 2. * self.friction / self.inverse_temperature) / \
        #        (1. + step_size / 2. * self.friction * self.inverse_mass)
        return torch.sqrt(step_size * 2. * self.friction / (self.inverse_temperature * self.inverse_mass)) / \
               (1. + step_size / 2. * self.friction)

    def midpoint_euler_step(self, p_in, step_size=None, a=None, b=None):
        """
        midpoint Euler step
        :param p_in: input value of p (the momentum)
        :param step_size: step size to be used
            default: self.half_step_size
        :param a: A-matrix to be used
            default: computed from step_size (or default if step_size is None)
        :param b: B-matrix to be used
            default: computed from step_size (or default if step_size is None)
        :return: p_out

        NB assumes all matrices to be diagonal.
        """
        a = a if a is not None else self._a(step_size)
        b = b if b is not None else self._b(step_size)

        noise = torch.normal(torch.zeros_like(p_in)).to(p_in.device)

        return a * p_in + b * noise

    def verlet_step(self, p_quarter, q_in, step_size=None):
        """
        Verlet scheme step
        :param p_quarter: p^{n+1/4}
        :param q_in: q^n
        :param step_size: step size to be used
            default: self.step_size
        :return: p_three_quarters_proposed, q_proposed
        """
        if step_size is None:
            step_size = self.step_size
            half_step_size = self.half_step_size
        else:
            half_step_size = step_size/2

        p_half = p_quarter - half_step_size * self.nabla_energy(q_in)
        q_proposed = q_in + step_size * self.inverse_mass * p_half
        p_three_quarters_proposed = p_half - half_step_size * self.nabla_energy(q_proposed)

        return p_three_quarters_proposed, q_proposed

    def acceptance_probability(self, q_in, p_quarter, q_proposed, p_three_quarters_proposed):
        """
        computes the acceptance probability in rejection step
        :param q_in: q^n
        :param p_quarter: p^{n+1/4}
        :param q_proposed: \tilde{q}^{n+1}
        :param p_three_quarters_proposed: \tilde{p}^{n+3/4}
        :return:
        """
        return torch.minimum(
            torch.as_tensor(1.),
            torch.exp(-self.inverse_temperature*(
                self.hamiltonian(q_proposed, p_three_quarters_proposed)
                - self.hamiltonian(q_in, p_quarter)
            ))
        ).squeeze(-1)

    def step(self, q_in, p_in, step_size=None):
        """
        takes one step with the full scheme
        :param q_in: q^n
        :param p_in: p^n
        :param step_size: step size to be used
            default: self.step_size
        :return: q^{n+1}, p^{n+1}
        """
        batch_size = q_in.shape[0]

        half_step_size = None if step_size is None else step_size/2.

        p_quarter = self.midpoint_euler_step(p_in, half_step_size)

        p_three_quarters_proposed, q_proposed = self.verlet_step(p_quarter, q_in, step_size)

        # accept/reject proposal (Metropolis-Hastings)
        random_noise = torch.rand(size=(batch_size,)).to(q_in.device)
        acceptance_probability = self.acceptance_probability(q_in, p_quarter, q_proposed, p_three_quarters_proposed)
        acceptance = random_noise <= acceptance_probability
        acceptance = acceptance.unsqueeze(-1).repeat((1, self.dim))

        if self.track_acceptance_rate:
            self._acceptance_rates.append(acceptance.float().mean().item())

        p_three_quarters = torch.where(acceptance, p_three_quarters_proposed, -p_quarter)
        q_out = torch.where(acceptance, q_proposed, q_in)

        # #temporary
        # p_three_quarters, q_out = p_three_quarters_proposed, q_proposed

        p_out = self.midpoint_euler_step(p_three_quarters, half_step_size)

        return q_out, p_out

    def forward(self, batch_size, steps=None, step_size=None):
        """
        generate batch_size samples
        :param batch_size: number of samples
        :param steps: number of steps to be used
            default: self.steps
        :param step_size: step size to be used
            default: self.step_size
        :return: samples
        """
        initial_states = self.initial_distribution(batch_size)

        steps = steps if steps is not None else self.steps

        q, p = initial_states[:, :self.dim], initial_states[:, self.dim:]

        with torch.no_grad():
            for _ in range(steps):
                q, p = self.step(q, p, step_size)

        self.initial_distribution.update(torch.cat([q, p], dim=1))

        return q


class MALSampler(_BaseSampler):
    """
    Metropolis Adjusted Langevin algorithm
    See Algorithm 2.9 in 'Free energy computations - a mathematical perspective'
    """

    def __init__(self,
                 dim,
                 energy_function,
                 inverse_temperature,
                 initial_distribution=None,
                 step_size=1e-2,
                 steps=1000
                 ):
        super().__init__(energy_function)
        self.dim = dim
        self.register_buffer('inverse_temperature', new_tensor(inverse_temperature))
        if initial_distribution is None:
            self.initial_distribution = DiagonalGaussianDistribution(dim)
        else:
            self.initial_distribution = initial_distribution

        self.step_size = step_size
        self.steps = steps

    def euler_maruyama_step(self, q_in, step_size=None):
        """
        Take a single Euler-Maruyama step
        :param q_in: q^n
        :param step_size: \Delta t
        :return: \tilde{q}^{n+1}
        """
        step_size = step_size or self.step_size
        noise = torch.normal(torch.zeros_like(q_in)).to(q_in.device)
        return q_in - step_size * self.nabla_energy(q_in) + torch.sqrt(2*step_size/self.inverse_temperature) * noise

    def acceptance_probability(self, q_in, q_proposed, step_size=None):
        """
        compute acceptance probability according to formula 2.38 on page 88 of Free energy computations
        :param q_in: q^n
        :param q_proposed: \tilde{q}^{n+1}
        :param step_size: \Delta t
        :return: probability of accepting q_proposed
        """
        step_size = step_size or self.step_size
        # we write formula 2.38 in 'Free energy computations' (page 88) as a single exponential
        exp_inp = (
                -self.inverse_temperature * (self.energy_function(q_proposed) - self.energy_function(q_in))
                - self.inverse_temperature / (4. * step_size) * (
                        torch.sum(torch.square(q_in - q_proposed + step_size * self.nabla_energy(q_proposed)), dim=-1,
                                  keepdim=True)
                        - torch.sum(torch.square(q_proposed - q_in + step_size * self.nabla_energy(q_in)), dim=-1,
                                    keepdim=True)
                )
        )
        return torch.minimum(torch.as_tensor(1.), torch.exp(exp_inp))

    def step(self, q_in, step_size=None):
        """
        Take a single step in the sampler
        This consists of first proposing a new state according to the Euler-Maruyama scheme
        and then performing Metropolis Hastings
        :param q_in:
        :param step_size:
        :return:
        """
        # propose according to Euler-Maruyama
        q_proposed = self.euler_maruyama_step(q_in, step_size)

        # get acceptance probability from Metropolis-Hastings ratio
        acceptance_probability = self.acceptance_probability(q_in, q_proposed, step_size)

        # either accept or reject the proposal
        batch_size = q_in.shape[0]
        noise = torch.rand((batch_size, 1)).to(q_in.device)
        acceptance = noise <= acceptance_probability
        acceptance = acceptance.repeat((1, self.dim))

        q_out = torch.where(acceptance, q_proposed, q_in)
        return q_out

    def forward(self, batch_size, steps=None, step_size=None):
        steps = steps or self.steps

        with torch.no_grad():
            q = self.initial_distribution(batch_size)
            for _ in range(steps):
                q = self.step(q, step_size=step_size)

        self.initial_distribution.update(q)

        return q


# sampling from the corresponding stationary distribution to an energy landscape is done by first sampling from some initial distribution, and then running either a Generalised Hamiltonian Monte Carlo sampler,
# or a Metropolis Adjusted Langevin sampler
# One thing to note is that in general, the landscape might not be 'centred' around the origin. If we use a centred Gaussian as an initial distribution, this might result
# in a strong bias for one of the minima in our sampling procedure.
# Ways to deal with this:

# - manually choose a better initial distribution (this is in general not doable for the same reasons as why we're doing this entire exercise in the first place)
# - sample from a wide centred normal distribution and hope for the best (in some cases that might work if we use enough steps in our sampler, but in general, this might be a bad strategy)
# - encode a bunch of data using the corresponding encoder model, and use the encoded data as starting points for the mcmc sampler 

# the latter approach is probably best, so in our experiments we only use the EmpiricalDistribution class
class InitialDistribution(abc.ABC):

    @abc.abstractmethod
    def forward(self, batch_size, **kwargs):
        pass

    def update(self, final_samples):
        pass

    @property
    @abc.abstractmethod
    def device(self):
        """
        Should return the device the sampler is on
        :return:
        """
        pass


class DiagonalGaussianDistribution(nn.Module, InitialDistribution):
    """
    Diagonal Gaussian Distribution
    can be used as initial distribution for samplers
    """

    def __init__(self, dim, mean=0., stddev=2.):
        super().__init__()
        self.dim = dim

        self.register_buffer(
            'mean',
            mean * torch.ones((dim,)) if not isinstance(mean, torch.Tensor) else new_tensor(mean)
        )
        self.register_buffer(
            'stddev',
            stddev * torch.ones((dim,)) if not isinstance(stddev, torch.Tensor) else new_tensor(stddev)
        )

    def forward(self, batch_size, **kwargs):
        mean = self.mean.repeat(batch_size, 1)
        stddev = self.stddev.repeat(batch_size, 1)

        return torch.normal(mean, stddev).to(self.device)

    @property
    def device(self):
        return self.mean.device


class EmpiricalDistribution(InitialDistribution, nn.Module):
    """EmpiricalDistribution 
    empirical distribution to be used as initial distribution for the mcmc samplers
    """
    def __init__(self, data, add_momentum=False, replace=False):
        """ 
        :param data: the data for which we want to create the empirical distribution
        :param add_momentum: whether to sample a (gaussian) momentum component besides the locations --- useful for GHMC samplers
        :param replace: whether to allow for duplicates in sampling from the distribution
        """
        super().__init__()
        self.data = data.view(-1, data.shape[-1])  # flatten all but the last dimensions
        self.size = self.data.shape[0]
        self._device = self.data.device
        self.add_momentum = add_momentum
        self.replace = replace

    def forward(self, batch_size, device=None, replace=None, **kwargs):
        if device is None:
            device = self.device 
        if replace is None:
            replace = self.replace
        indices = torch.from_numpy(np.random.choice(self.size, size=batch_size, replace=replace))
        result = torch.index_select(self.data, 0, indices).to(device=device)
        if self.add_momentum:
            momentum = torch.randn_like(result)
            result = torch.cat([result, momentum], -1)
        return result 

    @property
    def device(self):
        return self._device

    def to(self, *args, device=None, move_data=False, **kwargs):
        self._device = device 
        if move_data:
            self.data = self.data.to(*args, device=device, **kwargs)
        return self


# next, we make a function to perform gradient flow for an energy landscape
def flow(
    vector_field: nn.Module,
    initial_data: torch.Tensor,
    steps: int,
    step_size: float=1e-2,
    backward: bool=False,
    ):
    data = initial_data
    if not backward:
        for _ in range(steps):
            data = data + step_size * vector_field(data)
    else:
        for _ in range(steps):
            data = data - step_size * vector_field(data)
    return data 


def gradient_flow(
    energy_landscape: nn.Module,
    initial_data: torch.Tensor,
    steps: Union[int, None] = None,
    eps: float = 1e-1,
    step_size: float = 1e-2,
    max_steps=None
    ):
    """ 
    :param energy_landscape: torch module implementing the energy landscape. 
        is required to have a quadratic_term property or attribute consisting of a scalar tensor
        NB a side-effect of this function is that this module is set to eval mode
    :param initial_data: starting points from which to perform gradient descent
    :param steps: number of gd steps to perform. If None, the number of steps is computed from eps
    :param eps: if steps is None, the number of steps is computed from eps in the following way:
        if we assume that all 'noise directions' are gaussian with variance 1/energy_landscape.quadratic_term, 
        and we want all starting positions (in those directions) further away from zero than 3*standard deviation to get within eps from zero,
        then we pick steps to make that happen based on dz/dt = -c*z, z(0) = 3*sqrt(1/c), T is such that z(T) = eps  (here c = energy_landscape.quadratic_term)
        (so then steps = ceil(T/step_size))
    :param step_size: learning rate in gradient descent algorithm
    :param max_steps: maximum number of steps to perform. Ignored if steps is not None. No maximum is used if max_steps is None
    """
    vector_field = GradNet(energy_landscape)
    energy_landscape.eval()
    
    if steps is None:
        quadratic_coefficient = energy_landscape.quadratic_term.item()
        final_time = -1/quadratic_coefficient * (np.log(eps) + .5*np.log(quadratic_coefficient) - np.log(3))
        steps = int(np.ceil(final_time / step_size)) 
        if max_steps is not None:
            steps = min(max_steps, steps)
        print(f'using {steps} steps')

    return flow(vector_field, initial_data, steps, step_size=step_size, backward=True)


