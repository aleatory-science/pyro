from functools import partial
from time import time
from typing import Tuple

import matplotlib.pyplot as plt
import torch
from torch import tensor

from examples.lv_extension.lotka_volterra_func_ocl import run_lv
from pyro import sample, plate
from pyro.distributions import HalfNormal, Normal
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDelta
from pyro.optim import ClippedAdam

NOISE_SCALE = .3

print(torch.cuda.is_available())


def euler_method(fn, step_size, num_steps, state,
                 *args,
                 **kwargs):
    current_state = state
    fn = partial(fn, *args, **kwargs)
    res = torch.empty((num_steps, *state.shape))
    for i in range(num_steps):
        next_state = current_state + step_size * fn(current_state)
        res[i] = next_state
        current_state = next_state
    return res


def runge_kutta(fn, step_size, num_steps, init_prey, init_predator, *args):
    current_state = torch.stack((init_prey, init_predator), dim=-1)
    accum = torch.empty((num_steps, *current_state.shape))

    for i in range(num_steps):
        k1 = fn(current_state, *args)
        k2 = fn(current_state + step_size * k1 / 2, *args)
        k3 = fn(current_state + step_size * k2 / 2, *args)
        k4 = fn(current_state + step_size * k3, *args)
        current_state = current_state + step_size / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        accum[i] = current_state
    return accum


def lv_step(state: Tuple[float, float],  # prey, predator
            prey_growth_rate: float,  # prey_growth_rate
            predation_rate: float,  # predation_rate
            predator_growth_rate: float,  # predator_growth_rate
            predator_decline_rate: float  # predator_decline_rate:
            ):
    """ Lotka-Volterra equations

    prey/dt = prey_growth_rate * prey - predation_rate * prey * predator
    predator/dt = predator_growth_rate * prey * predator - predator_decline_rate * predator
    """
    prey = state[..., 0]
    predator = state[..., 1]
    dprey = (prey_growth_rate - predation_rate * predator) * prey
    dpredator = (predator_growth_rate * prey - predator_decline_rate) * predator
    return torch.stack((dprey, dpredator), dim=-1)


def model(times, obs, step_size=torch.tensor([.3]), num_steps=torch.tensor([300], dtype=torch.int64)):
    n, _, _ = obs.shape
    prior_dist = HalfNormal(2.)

    prey_init = sample('prey_init', prior_dist)
    predator_init = sample('predator_init', prior_dist)

    prey_growth_rate = sample('prey_growth_rate', prior_dist)
    predation_rate = sample('predation_rate', prior_dist)
    predator_growth_rate = sample('predator_growth_rate', prior_dist)
    predator_decline_rate = sample('predator_decline_rate', prior_dist)

    pp_res = run_lv(
        step_size, num_steps,
        prey_init,
        predator_init,
        prey_growth_rate,
        predation_rate,
        predator_growth_rate,
        predator_decline_rate
    )
    with plate('obs_plate', n, dim=-3):
        sample('obs', Normal(pp_res[times][None, :, :], NOISE_SCALE), obs=obs)


if __name__ == '__main__':
    system = dict(
        step_size=.1,
        num_step=300,
        init_prey=1.,
        init_predator=1.,  # prey, predator
        growth_prey=.45,
        predation=.4,
        growth_predator=.1,
        decline_predator=.2
    )
    torch.manual_seed(37)
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    steps = 1000

    system = {k: tensor(v) if k != 'num_steps' else tensor(v, dtype=torch.int64) for k, v in system.items()}

    pp_res = torch.tensor(run_lv(*system.values()))
    obs_times = tensor([28, 78, 128, 173, 228, 278], dtype=torch.long)
    obs = pp_res[obs_times]
    data = (obs_times, obs + Normal(0, NOISE_SCALE, ).sample((1000, *obs.shape,)))

    plt.plot(pp_res[:, 0], label='prey')
    plt.plot(pp_res[:, 1], '--', label='predator')
    plt.scatter(obs_times.repeat(1000, 1).T, torch.transpose(data[1][..., 0], 0, 1), color='blue', alpha=.01)
    plt.scatter(obs_times.repeat(1000, 1).T, torch.transpose(data[1][..., 1], 0, 1), color='orange', alpha=.01)
    plt.scatter(obs_times, obs[:, 0], marker='x', color='black', label='obs prey')
    plt.scatter(obs_times, obs[:, 1], marker='o', color='black', label='obs predator')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.show()

    x = torch.linspace(0., 10., 1000)
    plt.plot(x[1:], torch.exp(HalfNormal(2.).log_prob(x[1:])))
    plt.title('HalfNormal(2.)')
    plt.ylabel('ProbDens(x)')
    plt.xlabel('x')
    plt.show()
    x = torch.linspace(-5 * NOISE_SCALE, 5 * NOISE_SCALE, 1000)
    plt.plot(x, torch.exp(Normal(0, NOISE_SCALE).log_prob(x)))
    plt.title(f'Normal(0.,{NOISE_SCALE})')
    plt.ylabel('ProbDens(x)')
    plt.xlabel('x')
    plt.show()

    for v in data:
        v.to(dev)

    guide = AutoDelta(model)
    svi = SVI(model, guide, ClippedAdam({'lr': 1e-4}), Trace_ELBO())
    losses = []
    start = time()
    for step in range(steps):
        loss = svi.step(*data)
        losses.append(loss)
    print(f'Time take: {time() - start}')
    plt.plot(losses)
    plt.title('Losses')
    plt.ylabel('Loss')
    plt.xlabel('Step')
    plt.show()
