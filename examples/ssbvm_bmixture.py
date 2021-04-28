import pickle

import torch
from pyro.distributions import HalfNormal, VonMises
import matplotlib.pyplot as plt

import pyro
from pyro.distributions import (
    Beta,
    Categorical,
    Gamma,
    SineBivariateVonMises,
    SineSkewed,
    Uniform,
)
from pyro.infer import MCMC, NUTS


def model(obs, num_mix_comp=25):
    mix_weight_vals = pyro.sample('mix_weight_vals', Gamma(1. / num_mix_comp, 1.).expand((num_mix_comp,)))  # BDA p. 536
    mix_weights = mix_weight_vals / mix_weight_vals.sum()
    with pyro.plate('mixture', num_mix_comp):
        # BvM priors
        phi_loc = pyro.sample('phi_loc', VonMises(0., 1.))
        psi_loc = pyro.sample('psi_loc', VonMises(0., 1.))
        phi_conc = pyro.sample('phi_conc', HalfNormal(2.))
        psi_conc = pyro.sample('psi_conc', HalfNormal(2.))
        corr_scale = pyro.sample('corr', Beta(2., 2.))

        # SS prior
        skewness = torch.empty((num_mix_comp, 2)).view(-1, 2)
        tots = torch.zeros(num_mix_comp).view(-1)
        for i in range(2):
            skewness[..., i] = pyro.sample(f'skew{i}', Uniform(0., 1 - tots))
            tots += skewness[..., i]
        sign = pyro.sample('sign', Uniform(0., torch.ones((2,))).to_event(1))
        skewness = torch.where(sign < .5, -skewness, skewness)

    with pyro.plate('data', obs.size(-2)):
        assign = pyro.sample('mix_comp', Categorical(mix_weights), )
        bvm = SineBivariateVonMises(phi_loc=phi_loc[assign], psi_loc=psi_loc[assign],
                                    phi_concentration=phi_conc[assign], psi_concentration=psi_conc[assign],
                                    weighted_correlation=corr_scale[assign])
        pyro.sample('obs', SineSkewed(bvm, skewness[assign]), obs=obs)


def fetch_dihedrals(split='train'):
    # format one_hot(aa) + phi_angle + psi_angle
    data = pickle.load(open('data/9mer_fragments_processed.pkl', 'rb'))[split]['sequences'][..., -2:]
    return torch.tensor(data).view(-1, 2)


def fetch_toy_dihedrals(split='train'):
    # only 5 examples
    data = pickle.load(open('data/9mer_fragments_processed_toy.pkl', 'rb'))[split]['sequences'][..., -2:]
    return torch.tensor(data).view(-1, 2)


def main(show_viz=False):
    data = fetch_toy_dihedrals()

    if show_viz:
        plt.scatter(*data.T, alpha=.01, s=20)
        plt.show()
        plt.clf()

    kernel = NUTS(model)
    mcmc = MCMC(kernel, 1, 0)
    mcmc.run(data[:10])


def make_toy():
    data = pickle.load(open('data/9mer_fragments_processed.pkl', 'rb'))
    toy_data = {k: {kk: vv[:5] for kk, vv in v.items()} for k, v in data.items()}
    pickle.dump(toy_data, open('data/9mer_fragments_processed_toy.pkl', 'wb'))


if __name__ == '__main__':
    main()
