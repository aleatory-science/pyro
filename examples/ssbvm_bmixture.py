import matplotlib.pyplot as plt
import torch
from torch.distributions import VonMises, HalfNormal

import pyro
from pyro.distributions import Beta, Uniform, Dirichlet, Categorical
from pyro.distributions.bivariate_von_mises import SineBivariateVonMises, SineSkewed
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDelta


def model(obs, num_mix_comp=10):
    mix_weights = pyro.sample('mix_weights', Dirichlet(.5 * torch.ones(num_mix_comp)))
    with pyro.plate('mixture', num_mix_comp):
        locs = pyro.sample('locs', VonMises(0., 2 * torch.ones(2)))
        corr_scale = pyro.sample('corr', Beta(2., 2.))
        conc = pyro.sample('conc', HalfNormal(1.))
        skewness_val = pyro.sample('skewness_val', Uniform(-torch.ones(2), torch.ones(2)))
        skewness_scale = pyro.sample('skewness_scale', Beta(2., 2.))
        skewness = pyro.deterministic('skewness', skewness_scale * skewness_val / skewness_val.abs().sum())
    with pyro.plate('obs', obs.size(-2)):
        assign = pyro.sample('mix_comp', Categorical(mix_weights))
        bvm = SineBivariateVonMises(phi_loc=locs[assign, 0], psi_loc=locs[assign, 1],
                                    phi_concentration=conc[assign, 0], psi_concentration=conc[assign, 1],
                                    weighted_correlation=corr_scale[assign])
        pyro.sample('obs', SineSkewed(bvm, skewness[assign]), obs=obs)


def fetch_data():
    pass


def main():
    pyro.clear_param_store()
    adam = pyro.optim.Adam({"lr": .01})
    svi = SVI(model, AutoDelta(model), adam, loss=Trace_ELBO())
    data = fetch_data()

    losses = []
    for step in range(1000):
        losses.append(svi.step(data))

    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    main()
