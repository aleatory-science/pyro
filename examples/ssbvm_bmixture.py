import pickle
from functools import partial

import torch
import matplotlib.pyplot as plt
from torch import no_grad

import pyro
from pyro import poutine
from pyro.distributions import (
    Beta,
    Categorical,
    Normal,
    HalfNormal,
    VonMises,
    SineBivariateVonMises,
    SineSkewed,
    Uniform, Dirichlet,
)
from pyro.infer import MCMC, NUTS, config_enumerate, Predictive


@config_enumerate
def model(num_mix_comp=15):
    mix_weights = pyro.sample('mix_weights', Dirichlet(torch.ones((num_mix_comp,))))

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

        assert skewness.shape == (num_mix_comp, 2)

    with pyro.plate('obs_plate'):
        assign = pyro.sample('mix_comp', Categorical(mix_weights), )
        bvm = SineBivariateVonMises(phi_loc=phi_loc[assign], psi_loc=psi_loc[assign],
                                    phi_concentration=phi_conc[assign], psi_concentration=psi_conc[assign],
                                    weighted_correlation=corr_scale[assign])
        return pyro.sample('phi_psi', SineSkewed(bvm, skewness[assign]))


def cmodel(angles, num_mix_comp=15):
    poutine.condition(model, data={'phi_psi': angles})(num_mix_comp)


def dummy_model(num_mix_comp=2):
    mix_weights = pyro.sample('mix_weights', Dirichlet(torch.ones((num_mix_comp,))))
    with pyro.plate('mixture', num_mix_comp):
        scale = pyro.sample('scale', HalfNormal(1.).expand((2,)).to_event(1))
        locs = pyro.sample('locs', Normal(0., 1.).expand((2,)).to_event(1))
    with pyro.plate('data'):
        assign = pyro.sample('mix_comp', Categorical(mix_weights))
        pyro.sample('phi_psi', Normal(locs[assign], scale[assign]).to_event(1))


def cdummy_model(angles, num_mix_comp=5):
    poutine.condition(dummy_model, data={'phi_psi': angles})(num_mix_comp)


def fetch_dihedrals(split='train', subsample_to=50_000):
    # format one_hot(aa) + phi_angle + psi_angle
    assert subsample_to > 2
    data = pickle.load(open('data/9mer_fragments_processed.pkl', 'rb'))[split]['sequences'][..., -2:]
    data = torch.tensor(data).view(-1, 2).type(torch.float)
    subsample_to = min(data.shape[0], subsample_to)
    perm = torch.randint(0, data.shape[0] - 1, size=(subsample_to,))
    return data[perm]


def fetch_toy_dihedrals(split='train', *args, **kwargs):
    # only 45 examples
    data = pickle.load(open('data/9mer_fragments_processed_toy.pkl', 'rb'))[split]['sequences'][..., -2:]
    return torch.tensor(data).view(-1, 2).type(torch.float)


def main(num_samples=10, show_viz=False):
    data = fetch_toy_dihedrals(subsample_to=1000)

    if show_viz:
        ramachandran_plot(data)

    kernel = NUTS(cdummy_model)
    mcmc = MCMC(kernel, num_samples)
    mcmc.run(data)
    if show_viz:
        mcmc.summary()
    post_samples = mcmc.get_samples()

    predictive = Predictive(dummy_model, post_samples, return_sites=('phi_psi',))

    pred_data = predictive()

    if show_viz:
        ramachandran_plot(pred_data['phi_psi'], 'pred', color='orange')
    for _ in range(10):
        pred_data = predictive()

        if show_viz:
            ramachandran_plot(pred_data['phi_psi'], None, color='orange')
    plt.show()


def ramachandran_plot(data, label='ground_truth', color='blue'):
    plt.scatter(*data.T, alpha=.5, s=20, label=label, color=color)
    plt.xlabel('phi')
    plt.ylabel('psi')
    plt.title('Ramachandran plot')
    plt.legend()


def make_toy():
    data = pickle.load(open('data/9mer_fragments_processed.pkl', 'rb'))
    toy_data = {k: {kk: vv[:5] for kk, vv in v.items()} for k, v in data.items()}
    pickle.dump(toy_data, open('data/9mer_fragments_processed_toy.pkl', 'wb'))


if __name__ == '__main__':
    main(show_viz=True)
