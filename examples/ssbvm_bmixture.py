import logging
import pickle
import sys
from math import pi
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

import pyro
from pyro import poutine
from pyro.distributions import (
    Beta,
    Categorical,
    Dirichlet,
    Gamma,
    SineBivariateVonMises,
    SineSkewed,
    Uniform,
    VonMises, )
from pyro.infer import MCMC, NUTS, Predictive, config_enumerate
from pyro.infer.autoguide import init_to_median
from pyro.infer.mcmc.util import _predictive_sequential
from pyro.ops.stats import waic
from tests.common import tensors_default_to

logging.getLogger('matplotlib.font_manager').disabled = True

AMINO_ACIDS = ['M', 'N', 'I', 'F', 'E', 'L', 'R', 'D', 'G', 'K', 'Y', 'T', 'H', 'S', 'P', 'A', 'V', 'Q', 'W', 'C']


def _drop_skew_params(dictionary):
    return {k: v for k, v in dictionary.items() if 'skew' not in k}


@config_enumerate
def sine_model(num_mix_comp=2):
    # Mixture prior
    mix_weights = pyro.sample('mix_weights', Dirichlet(torch.ones((num_mix_comp,))))

    # Hprior BvM
    # Bayesian Inference and Decision Theory by Kathryn Blackmond Laskey
    beta_mean_phi = pyro.sample('beta_mean_phi', Uniform(0., 1.))
    beta_prec_phi = pyro.sample('beta_prec_phi', Gamma(1., 1. / 20.))  # shape, rate
    halpha_phi = beta_mean_phi * beta_prec_phi
    beta_mean_psi = pyro.sample('beta_mean_psi', Uniform(0, 1.))
    beta_prec_psi = pyro.sample('beta_prec_psi', Gamma(1., 1. / 20.))  # shape, rate
    halpha_psi = beta_mean_psi * beta_prec_psi

    with pyro.plate('mixture', num_mix_comp):
        # BvM priors
        phi_loc = pyro.sample('phi_loc', VonMises(pi, 2.))
        psi_loc = pyro.sample('psi_loc', VonMises(-pi / 2, 2.))
        phi_conc = pyro.sample('phi_conc', Beta(halpha_phi, beta_prec_phi - halpha_phi))
        psi_conc = pyro.sample('psi_conc', Beta(halpha_psi, beta_prec_psi - halpha_psi))
        corr_scale = pyro.sample('corr_scale', Beta(2., 5.))

    with pyro.plate('obs_plate'):
        assign = pyro.sample('mix_comp', Categorical(mix_weights), )
        bvm = SineBivariateVonMises(phi_loc=phi_loc[assign], psi_loc=psi_loc[assign],
                                    phi_concentration=1000 * phi_conc[assign],
                                    psi_concentration=1000 * psi_conc[assign],
                                    weighted_correlation=corr_scale[assign])
        return pyro.sample('phi_psi', bvm)


@config_enumerate
def ss_model(num_mix_comp=2):
    # Mixture prior
    mix_weights = pyro.sample('mix_weights', Dirichlet(torch.ones((num_mix_comp,))))

    # Hprior BvM
    # Bayesian Inference and Decision Theory by Kathryn Blackmond Laskey
    beta_mean_phi = pyro.sample('beta_mean_phi', Uniform(0., 1.))
    beta_prec_phi = pyro.sample('beta_prec_phi', Gamma(1., 1 / 20.))  # shape, rate
    halpha_phi = beta_mean_phi * beta_prec_phi
    beta_mean_psi = pyro.sample('beta_mean_psi', Uniform(0, 1.))
    beta_prec_psi = pyro.sample('beta_prec_psi', Gamma(1., 1 / 20.))  # shape, rate
    halpha_psi = beta_mean_psi * beta_prec_psi

    with pyro.plate('mixture', num_mix_comp):
        # BvM priors
        phi_loc = pyro.sample('phi_loc', VonMises(pi, 2.))
        psi_loc = pyro.sample('psi_loc', VonMises(-pi / 2, 2.))
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
        assign = pyro.sample('mix_comp', Categorical(mix_weights), infer={"enumerate": "parallel"})
        bvm = SineBivariateVonMises(phi_loc=phi_loc[assign], psi_loc=psi_loc[assign],
                                    phi_concentration=1000 * phi_conc[assign],
                                    psi_concentration=1000 * psi_conc[assign],
                                    weighted_correlation=corr_scale[assign])
        return pyro.sample('phi_psi', SineSkewed(bvm, skewness[assign]))


def cmodel(model, angles, num_mix_comp=2):
    poutine.condition(model, data={'phi_psi': angles})(num_mix_comp)


def fetch_aa_dihedrals(split='train', subsample_to=1000_000):
    data = pickle.load(open('data/9mer_fragments_processed.pkl', 'rb'))[split]['sequences']
    data_aa = np.argmax(data[..., :20], -1)
    data = {aa: data[..., -2:][data_aa == i] for i, aa in enumerate(AMINO_ACIDS)}
    [np.random.shuffle(v) for v in data.values()]
    data = {aa: aa_data[:min(subsample_to, aa_data.shape[0])] for aa, aa_data in data.items()}
    data = {aa: torch.tensor(aa_data, dtype=torch.float) for aa, aa_data in data.items()}
    return data


def run_hmc(model, data, num_mix_comp, num_samples):
    kernel = NUTS(cmodel, init_strategy=init_to_median(), max_plate_nesting=1)
    mcmc = MCMC(kernel, num_samples, num_samples // 3)
    mcmc.run(model, data, num_mix_comp)
    mcmc.summary()
    post_samples = mcmc.get_samples()
    return post_samples


def compute_waic(samples, num_samples, model_args: Sequence = tuple(), model_kwargs={}):
    trs = _predictive_sequential(cmodel, samples, model_args, model_kwargs, num_samples, tuple(),
                                 return_trace=True)  # FIXME
    [tr.compute_log_prob(lambda name, _: name == 'phi_psi') for tr in trs]
    ic, neff = waic(torch.stack([tr.nodes['phi_psi']['log_prob'] for tr in trs], dim=0).mean(0))
    return {'waic': ic, 'waic_neff': neff}


def compute_log_likelihood(samples, num_samples, model_args: Sequence = tuple(), model_kwargs={}):
    traces = _predictive_sequential(cmodel, samples, model_args, model_kwargs, num_samples, tuple(),
                                    return_trace=True)
    [tr.compute_log_prob(lambda name, _: name == 'phi_psi') for tr in traces]
    return torch.stack([tr.nodes['phi_psi']['log_prob_sum'] for tr in traces], dim=0).mean(0)


def sample_posterior_predicitve(model, posterior_samples, num_mix_comp):
    predictive = Predictive(model, posterior_samples, return_sites=('phi_psi',))
    pred_data = []
    fail = 0
    for _ in range(2):  # TODO: parallelize
        try:
            pred_data.append(predictive(num_mix_comp)['phi_psi'].squeeze())
        except Exception as e:
            print(e)
            fail += 1
    print(f'failed samples {fail}')
    return torch.stack(pred_data).view(-1, 2).to('cpu')


def kde_ramachandran_plot(pred_data, data, file_name='kde_rama.png'):
    fig, axs = plt.subplots(1, len(pred_data))

    for i, aa in enumerate(pred_data.keys()):
        plt.scatter(data[aa][:, 0], data[aa][:, 1], alpha=.01, color='k')
        sns.kdeplot(pred_data[aa][:, 0].numpy(),
                    pred_data[aa][:, 1].numpy(),
                    label='Predictions',
                    cmap='coolwarm',
                    ax=axs[i])
        axs[i].set_xlabel('phi')
    axs[0].set_ylabel('psi')
    plt.title('Ramachandran plot')
    plt.legend(loc='upper right')
    if file_name:
        viz_dir = Path(__file__).parent.parent / 'viz'
        viz_dir.mkdir(exist_ok=True)
        plt.savefig(str(viz_dir / file_name))
    else:
        plt.show()
    plt.clf()


def main(num_mix_start=20, num_mix_end=50, num_samples=250, aas=('S', 'P', 'G'),
         use_cuda=False, capture_std=True, rerun_inference=False):
    if torch.cuda.is_available() and use_cuda:
        device_context = tensors_default_to("cuda")
    else:
        device_context = tensors_default_to("cpu")

    with device_context:
        data = fetch_aa_dihedrals(subsample_to=200)
    for aa in aas:
        for num_mix_comp in range(num_mix_start, num_mix_end):

            if capture_std:
                out_dir = Path( __file__).parent / "runs" / "outs" / aa
                out_dir.mkdir(exist_ok=True)
                sys.stdout = (out_dir/ f'ssbvm_bmixture_aa{aa}_comp{num_mix_comp}_steps{num_samples}.out').open('w')

            pkl_dir = Path(__file__).parent / "runs" / 'pkls' / aa
            pkl_dir.mkdir(exist_ok=True)
            chain_file = pkl_dir / f'ssbvm_bmixture_aa{aa}_comp{num_mix_comp}_steps{num_samples}.pkl'

            if torch.cuda.is_available() and use_cuda:
                device_context = tensors_default_to("cuda")
            else:
                device_context = tensors_default_to("cpu")
            with device_context:
                if rerun_inference or not chain_file.exists():
                    posterior_samples = {aa: {'ss': run_hmc(ss_model, data[aa], num_mix_comp, num_samples),
                                              'sine': run_hmc(sine_model, data[aa], num_mix_comp, num_samples)}}
                    pickle.dump(posterior_samples, chain_file.open('wb'))
                else:
                    print(pickle.load(chain_file.open('rb')))

            if capture_std:
                sys.stdout.close()


if __name__ == '__main__':
    main(use_cuda=True, rerun_inference=True, capture_std=True)
