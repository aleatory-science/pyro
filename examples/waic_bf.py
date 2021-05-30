import logging
import pickle
from collections import defaultdict
from math import pi
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

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
    VonMises,
)
from pyro.infer import config_enumerate
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
        psi_loc = pyro.sample('psi_loc', VonMises(0., .1))
        phi_conc = pyro.sample('phi_conc', Beta(halpha_phi, beta_prec_phi - halpha_phi))
        psi_conc = pyro.sample('psi_conc', Beta(halpha_psi, beta_prec_psi - halpha_psi))
        corr_scale = pyro.sample('corr_scale', Beta(2., 5.))

    with pyro.plate('obs_plate'):
        assign = pyro.sample('mix_comp', Categorical(mix_weights), )
        bvm = SineBivariateVonMises(phi_loc=phi_loc[assign], psi_loc=psi_loc[assign],
                                    phi_concentration=150 * phi_conc[assign],
                                    psi_concentration=150 * psi_conc[assign],
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
        psi_loc = pyro.sample('psi_loc', VonMises(0., .1))
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


def cmodel(model, angles, num_mix_comp=2):
    poutine.condition(model, data={'phi_psi': angles})(num_mix_comp)


def compute_waic(samples, num_samples, model_args: Sequence = tuple(), model_kwargs={}):
    trs = _predictive_sequential(cmodel, samples, model_args, model_kwargs, num_samples, tuple(),
                                 return_trace=True)  # FIXME
    [tr.compute_log_prob(lambda name, _: name == 'phi_psi') for tr in trs]
    ic, _ = waic(torch.stack([tr.nodes['phi_psi']['log_prob'] for tr in trs], dim=0).mean(0))
    return ic


def fetch_aa_dihedrals(split='train', subsample_to=1000_000):
    data = pickle.load(open('data/9mer_fragments_processed.pkl', 'rb'))[split]['sequences']
    data_aa = np.argmax(data[..., :20], -1)
    data = {aa: data[..., -2:][data_aa == i] for i, aa in enumerate(AMINO_ACIDS)}
    [np.random.shuffle(v) for v in data.values()]
    data = {aa: aa_data[:min(subsample_to, aa_data.shape[0])] for aa, aa_data in data.items()}
    data = {aa: torch.tensor(aa_data, dtype=torch.float) for aa, aa_data in data.items()}
    return data


def plot_waic(waics):
    fig, axs = plt.subplots(len(waics),1)
    for (aa, mwaic), ax in zip(waics.items(), axs):
        for model, waic in mwaic.items():
            ax.plot(np.arange(len(waic)) + 3, np.array(waic).reshape(-1), label=model)
        ax.set_xlabel(aa)
        ax.yaxis.set_ticks([])
        ax.set_ylabel('WAIC')

    plt.legend()
    plt.show()




def main(num_samples=1500, aas=('G', 'S', 'P'), use_cuda=False):
    if torch.cuda.is_available() and use_cuda:
        device_context = tensors_default_to("cuda")
        device = "cuda"
    else:
        device_context = tensors_default_to("cpu")
        device = "cpu"


    with device_context:
        data = fetch_aa_dihedrals(subsample_to=100)
        # val_data = fetch_aa_dihedrals(split='valid')
        waics = {}
        for aa in aas:
            ss_waics = []
            sine_waics = []
            for num_mix_comp in tqdm.tqdm(range(3, 45)):  # 45
                chain_file = Path(
                    __file__).parent / f'steps_1500/ssbvm_bmixture_comp{num_mix_comp}_steps1500.pkl'

                posterior_samples = {k: {kk: {kkk: vvv.detach().to(device) for kkk, vvv in vv.items()}
                                         for kk, vv in v.items()}
                                     for k, v in pickle.load(chain_file.open('rb')).items()}

                ss_waics.append(compute_waic(posterior_samples[aa]['ss'], num_samples,
                                             model_args=(ss_model, data[aa], num_mix_comp)))
                sine_waics.append(compute_waic(posterior_samples[aa]['sine'], num_samples,
                                         model_args=(sine_model, data[aa], num_mix_comp)))
            waics[aa] = {'ss': ss_waics,  'sine': sine_waics}

    print({k: {kk: np.argmin(np.array(vv).reshape(-1))+3 for kk, vv in v.items()} for k,v in waics.items()})
    plot_waic(waics)

    # log_bf = compute_log_likelihood(posterior_samples[aa]['ss'],
    #                                 num_samples,
    #                                 model_args=(ss_model, val_data[aa], num_mix_comp)) -\
    #          compute_log_likelihood(posterior_samples[aa]['sine'],
    #                                 num_samples,
    #                                 model_args=(sine_model, val_data[aa], num_mix_comp))


if __name__ == '__main__':
    main()
