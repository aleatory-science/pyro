from math import pi

import numpy as np
import seaborn
import torch
from torch import tensor


def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den * x / number))
        com = gcd(num, den)
        (num, den) = (int(num / com), int(den / com))
        if den == 1:
            if num == 0:
                return r'$0$'
            if num == 1:
                return r'$%s$' % latex
            elif num == -1:
                return r'$-%s$' % latex
            else:
                return r'$%s%s$' % (num, latex)
        else:
            if num == 1:
                return r'$\frac{%s}{%s}$' % (latex, den)
            elif num == -1:
                return r'$\frac{-%s}{%s}$' % (latex, den)
            else:
                return r'$\frac{%s%s}{%s}$' % (num, latex, den)

    return _multiple_formatter


def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2 * np.pi) ** n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)

    return np.exp(-fac / 2) / N


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm, colors
    from sklearn.neighbors import KernelDensity

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    import math

    from pyro.distributions import VonMises, SineSkewed, Gamma, SineBivariateVonMises, Uniform, Beta
    phi_loc = tensor(-1.1326572895050049)
    psi_loc = tensor(2.5)
    phi_conc = tensor( 149.99998474121094)
    psi_conc = tensor(1.1610255241394043)
    corr = tensor(0.08815242350101471)

    plt.xlim([-pi, pi])
    plt.ylim([-pi, pi])
    data = SineBivariateVonMises(phi_loc, psi_loc, phi_conc, psi_conc, corr).sample((75_000,))
    plt.hexbin(data[:, 0], data[:, 1], cmap="Greys", extent=[-math.pi, math.pi, -math.pi, math.pi])

    plt.show()

    exit()

    x = torch.linspace(1e-10, 10, 100)
    plt.plot(x, Gamma(1., 1 / 20.).log_prob(x).exp())
    plt.show()

    exit()

    plt.bar(['Glycine', 'Serine', 'Proline'], [721.2739, 134.6505, -88.6289], color=['blue', 'blue', 'red'])
    plt.axhline(y=0., color='gray', linestyle='--')
    plt.ylabel('log Bayes factor')
    plt.show()

    exit()

    vm = VonMises(0., .5)
    ss = SineSkewed(vm, tensor([.5]))
    x = np.linspace(-math.pi, math.pi, 100)
    plt.plot(x, vm.log_prob(tensor(x)).exp())
    print(ss.log_prob(tensor(x).view(-1, 1)).shape)
    plt.plot(x, ss.log_prob(tensor(x).view(-1, 1)).view(-1).exp())
    plt.tight_layout()
    plt.savefig('vm.png', dpi=300, bbox_inches='tight', transparent=True)

    exit()
    vm = VonMises(0., .5)
    ss = SineSkewed(vm, tensor([.5]))
    x = np.linspace(-math.pi, math.pi, 100)
    plt.plot(x, vm.log_prob(tensor(x)).exp())
    print(ss.log_prob(tensor(x).view(-1, 1)).shape)
    plt.plot(x, ss.log_prob(tensor(x).view(-1, 1)).view(-1).exp())
    plt.tight_layout()
    plt.savefig('vm.png', dpi=300, bbox_inches='tight', transparent=True)

    exit()

    N = 60
    X = np.linspace(-3, 3, N)
    Y = np.linspace(-3, 4, N)
    X, Y = np.meshgrid(X, Y)

    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    mu = np.array([0., 0.])
    Sigma = np.array([[1., 0.], [0, 1.]])

    Z = multivariate_gaussian(pos, mu, Sigma)
    fig, ax = plt.subplots()
    cset = ax.contour(X, Y, Z, c='k', alpha=.5)
    Sigma = np.array([[1., .5], [.5, 1.]])
    Z = multivariate_gaussian(pos, mu, Sigma)
    cset = ax.contourf(X, Y, Z, cmap=cm.coolwarm, alpha=.9)

    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    ax.set_aspect('equal', 'box')
    plt.savefig('correlation.png', dpi=300, bbox_inches='tight', transparent=True)
    fig.tight_layout()
    plt.clf()
    exit()

    # Plot between -10 and 10 with .001 steps.
    x_axis = np.arange(-10, 10, 0.001)
    # Mean = 0, SD = 2.

    plt.plot(x_axis, norm.pdf(x_axis, 5, 1), c='orange')
    plt.plot(x_axis, norm.pdf(x_axis, 0, 1), c='b')
    plt.plot(x_axis, norm.pdf(x_axis, -5, 1), c='g')
    plt.axvline(x=5., color='orange', linestyle='--')
    plt.axvline(x=0., color='b', linestyle='--')
    plt.axvline(x=-5., color='g', linestyle='--')

    plt.tight_layout()
    plt.savefig('location.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.clf()
    exit()

    phi_loc = tensor(0.)
    psi_loc = tensor(-pi / 2)
    phi_conc = tensor(4.)
    psi_conc = tensor(1.)
    corr = tensor(.3)

    fig, ax = plt.subplots()
    bvm = SineBivariateVonMises(phi_loc, psi_loc, phi_conc, psi_conc, corr)
    data = SineBivariateVonMises(phi_loc, psi_loc, phi_conc, psi_conc, corr).sample((75_000,))
    kde = KernelDensity().fit(data)
    seaborn.kdeplot(data[:, 0], data[:, 1], levels=15, shade=True, shade_lowest=True, cmap="coolwarm", cbar=True)
    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'$\psi$')

    fig.tight_layout()
    plt.savefig('flat_torus_bvm.png', bbox_inches='tight', dpi=300, transparent=True)
    plt.clf()

    angle = np.linspace(0, 2 * np.pi, 120)
    theta, phi = np.meshgrid(angle, angle)
    r, R = 1., 1.
    X = (R + r * np.cos(phi)) * np.cos(theta)
    Y = (R + r * np.cos(phi)) * np.sin(theta)
    Z = r * np.sin(phi)

    colorfunction = kde.score_samples(np.vstack((X.ravel(), Y.ravel())).T).reshape(X.shape)

    # colorfunction = (X ** 2 + Y ** 2)
    norm = colors.Normalize(colorfunction.min(), colorfunction.max())

    r, R = .5, 1.
    X = (R + r * np.cos(phi)) * np.cos(theta)
    Y = (R + r * np.cos(phi)) * np.sin(theta)
    Z = r * np.sin(phi)
    # Display the mesh
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.grid(False)  # Hide axes ticks
    ax.axis('off')
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cm.coolwarm(norm(colorfunction)))

    fig.tight_layout()
    plt.savefig('torus_bvm.png', bbox_inches='tight', dpi=300, transparent=True)
