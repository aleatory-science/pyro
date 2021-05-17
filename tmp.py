from math import pi

import matplotlib.pyplot as plt
import seaborn
from torch import stack, tensor
from torch.distributions import VonMises

from pyro.distributions import SineBivariateVonMises, Uniform
from tests.common import assert_close

if __name__ == "__main__":
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import colors, cm
    from sklearn.neighbors import KernelDensity

    phi_loc = tensor(0.)
    psi_loc = tensor(-pi/ 2)
    phi_conc = tensor(4.)
    psi_conc = tensor(1.)
    corr = tensor(.3)

    data = SineBivariateVonMises(phi_loc, psi_loc, phi_conc, psi_conc, corr).sample((5_000,))
    kde = KernelDensity().fit(data)

    angle = np.linspace(0, 2 * np.pi, 32)
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
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cm.jet(norm(colorfunction)))

    fig.tight_layout()
    plt.show()
