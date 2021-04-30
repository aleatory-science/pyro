import pickle
from functools import partial
from pathlib import Path

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
def model(num_mix_comp=2):
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


def cmodel(angles, num_mix_comp=2):
    poutine.condition(model, data={'phi_psi': angles})(num_mix_comp)


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


def main(num_samples=1000, show_viz=False):
    num_mix_comp = 2
    data = fetch_toy_dihedrals(subsample_to=1000)

    kernel = NUTS(cmodel)
    mcmc = MCMC(kernel, num_samples, num_samples // 2)
    mcmc.run(data, num_mix_comp)
    if show_viz:
        mcmc.summary()
    post_samples = mcmc.get_samples()

    predictive = Predictive(model, post_samples, return_sites=('phi_psi',))


    pred_data = []
    for _ in range(5):
        try:
            pred_data.append(predictive(num_mix_comp)['phi_psi'].squeeze())
        except:
            pass

    pred_data = torch.stack(pred_data).view(-1, 2)
    if show_viz:
        ramachandran_plot(data, pred_data, file_name='')


def ramachandran_plot(obs, pred_data, file_name='rama.png'):
    plt.scatter(*pred_data.T, alpha=.1, s=20, label='pred', color='orange')
    plt.scatter(*obs.T, alpha=.5, s=20, label='ground_truth', color='blue')
    plt.legend()
    plt.xlabel('phi')
    plt.ylabel('psi')
    plt.title('Ramachandran plot')
    if file_name:
        viz_dir = Path(__file__).parent.parent / 'viz'
        viz_dir.mkdir(exist_ok=True)
        plt.savefig(str(viz_dir / file_name))
    else:
        plt.show()
    plt.clf()


def make_toy():
    data = pickle.load(open('data/9mer_fragments_processed.pkl', 'rb'))
    toy_data = {k: {kk: vv[:5] for kk, vv in v.items()} for k, v in data.items()}
    pickle.dump(toy_data, open('data/9mer_fragments_processed_toy.pkl', 'wb'))


if __name__ == '__main__':
    main(show_viz=True)
