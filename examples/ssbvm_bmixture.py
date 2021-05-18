import pickle
from pathlib import Path
from math import pi
import warnings
import logging
from time import time

import seaborn as sns

import torch
import matplotlib.pyplot as plt
from torch.distributions import AffineTransform

from pyro.infer.autoguide import init_to_sample, init_to_median
from tests.common import tensors_default_to

import pyro
from pyro import poutine
from pyro.distributions import (
    Beta,
    Categorical,
    HalfNormal,
    TransformedDistribution,
    VonMises,
    SineBivariateVonMises,
    SineSkewed,
    Uniform, Dirichlet, Gamma,
)
from pyro.infer import MCMC, NUTS, config_enumerate, Predictive

logging.getLogger('matplotlib.font_manager').disabled = True


@config_enumerate
def model(num_mix_comp=2):
    # Mixture prior
    mix_weights = pyro.sample('mix_weights', Dirichlet(torch.ones((num_mix_comp,))))

    # Hprior BvM
    # Bayesian Inference and Decision Theory by Kathryn Blackmond Laskey
    beta_mean_phi = pyro.sample('beta_mean_phi', Uniform(0., 1.))
    beta_prec_phi = pyro.sample('beta_prec_phi', Gamma(1., 1 / num_mix_comp))  # shape, rate
    halpha_phi = beta_mean_phi * beta_prec_phi
    beta_mean_psi = pyro.sample('beta_mean_psi', Uniform(0, 1.))
    beta_prec_psi = pyro.sample('beta_prec_psi', Gamma(1., 1 / num_mix_comp))  # shape, rate
    halpha_psi = beta_mean_psi * beta_prec_psi

    with pyro.plate('mixture', num_mix_comp):
        # BvM priors
        phi_loc = pyro.sample('phi_loc', VonMises(pi, 2.))
        psi_loc = pyro.sample('psi_loc', VonMises(-1.5 + pi, 2.))
        phi_conc = pyro.sample('phi_conc', Beta(halpha_phi, beta_prec_phi - halpha_phi))
        psi_conc = pyro.sample('psi_conc', Beta(halpha_psi, beta_prec_psi - halpha_psi))
        corr_scale = pyro.sample('corr_scale', Beta(2., 5.))

        # SS prior
        skew_phi = pyro.sample('skew_phi', Uniform(-1., 1.))
        psi_bound = 1 - skew_phi.abs()
        skew_psi = pyro.sample('skew_psi', Uniform(-1., 1.))
        skewness = torch.stack((skew_phi, psi_bound * skew_psi), dim=-1)
        assert skewness.shape == (num_mix_comp, 2)

    with pyro.plate('obs_plate'):
        assign = pyro.sample('mix_comp', Categorical(mix_weights), )
        bvm = SineBivariateVonMises(phi_loc=phi_loc[assign], psi_loc=psi_loc[assign],
                                    phi_concentration=150 * phi_conc[assign],
                                    psi_concentration=150 * psi_conc[assign],
                                    weighted_correlation=corr_scale[assign])
        return pyro.sample('phi_psi', SineSkewed(bvm, skewness[assign]))


def cmodel(angles, num_mix_comp=2):
    poutine.condition(model, data={'phi_psi': angles})(num_mix_comp)


def fetch_dihedrals(split='train', subsample_to=1000_000):
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


def main(num_samples=640, show_viz=False, use_cuda=False):
    num_mix_comp = 2  # expected between 20-50
    if torch.cuda.is_available() and use_cuda:
        device_context = tensors_default_to("cuda")
    else:
        device_context = tensors_default_to("cpu")

    with device_context:
        data = fetch_dihedrals(subsample_to=50_000)

        kernel = NUTS(cmodel, max_tree_depth=6, init_strategy=init_to_sample())
        mcmc = MCMC(kernel, num_samples, num_samples // 2)
        mcmc.run(data, num_mix_comp)
        mcmc.summary()
        post_samples = mcmc.get_samples()
        pickle.dump(post_samples, open(f'ssbvm_bmixture_comp{num_mix_comp}_steps{num_samples}_full.pkl', 'wb'))

    if show_viz:
        predictive = Predictive(model, post_samples, return_sites=('phi_psi',))
        pred_data = []
        fail = 0
        for _ in range(5):  # TODO: parallelize
            try:
                pred_data.append(predictive(num_mix_comp)['phi_psi'].squeeze())
            except Exception as e:
                print(e)
                fail += 1
        pred_data = torch.stack(pred_data).view(-1, 2).to('cpu')
        print(f'failed samples {fail}')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ramachandran_plot(data.to('cpu'), pred_data)
            kde_ramachandran_plot(pred_data)


def kde_ramachandran_plot(pred_data, data, file_name='kde_rama.png'):
    plt.scatter(data[:, 0], data[:, 1], alpha=.01, color='k')
    sns.kdeplot(pred_data[:, 0].numpy(), pred_data[:, 1].numpy(), label='Predictions', cmap='coolwarm')
    plt.legend(loc='upper right')
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


def ramachandran_plot(obs, pred_data, file_name='rama.png'):
    plt.scatter(*obs.T, alpha=.1, s=20, label='ground_truth', color='blue')
    plt.scatter(pred_data[:, 0], pred_data[:, 1], alpha=.5, s=20, label='pred', color='orange')
    plt.legend(loc='upper right')
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
    main(show_viz=True, use_cuda=True)
