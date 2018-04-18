__author__ = "jeremiah"

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import pickle as pk
import random

import numpy as np
import seaborn as sns

import GPy as gp
from src.kernel.spectral_mixture import SpectralMixture

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


fit_residual = False


# random sample location

n_site = 500
n_dim = 2
np.random.seed(100)
loc_site = np.random.normal(size=(n_site, n_dim))
noise = 0.01


# random sample response
def generate_pol(x, y, noise=0.1):
    y_true = 0.2 * x + 0.5 * y + np.sqrt(x**2 + y**2 + 5* np.cos(x*y)) + \
             np.sin(x) + np.cos(y) + np.logaddexp(x*y, x)
    y_true = np.atleast_2d(y_true).T

    y_obs = y_true + np.random.normal(scale=noise, size=y_true.shape)
    return y_obs, y_true


y_obs, y_true = generate_pol(x=loc_site[:, 0],
                             y=loc_site[:, 1],
                             noise=noise)


# generate base prediction
kerns = [gp.kern.Linear(n_dim), # + gp.kern.White(n_dim),
         gp.kern.Poly(n_dim, order=2), # + gp.kern.White(n_dim),
         gp.kern.Poly(n_dim, order=3), # + gp.kern.White(n_dim),
         gp.kern.Poly(n_dim, order=4), # + gp.kern.White(n_dim),
         gp.kern.RBF(n_dim, ARD=True), # + gp.kern.White(n_dim),
         gp.kern.OU(n_dim, ARD=True),  # + gp.kern.White(n_dim),
         gp.kern.Matern32(n_dim, ARD=True),  # + gp.kern.White(n_dim),
         gp.kern.Matern52(n_dim, ARD=True), # + gp.kern.White(n_dim),
         gp.kern.MLP(n_dim, ARD=True), # + gp.kern.White(n_dim),
         SpectralMixture(n_dim)]# + gp.kern.White(n_dim)]

# create simple GP model
tr_id = range(n_site/2)
cv_id = range(n_site/2, n_site)

loc_tr, loc_cv = loc_site[tr_id], loc_site[cv_id]
y_tr, y_cv = y_obs[tr_id], y_obs[cv_id]

pred_tr = np.zeros(shape=(len(y_tr), len(kerns)))
pred_cv = np.zeros(shape=(len(y_cv), len(kerns)))

model_dict = dict()

for k_id in range(len(kerns)):
    kern_name = ["Linear", "Poly2", "Poly3", "Poly4",
                 "RBF_ARD", "Matern_12_ARD", "Matern_32_ARD", "Matern_52_ARD",
                 "MLP_ARD", "SpecMix"][k_id]

    kernel = kerns[k_id]
    m = gp.models.GPRegression(X=loc_tr, Y=y_tr, kernel=kernel)
    m.optimize(messages=False, max_f_eval=1000)
    pred_tr[:, k_id] = m.predict(Xnew=loc_tr)[0].T
    pred_cv[:, k_id] = m.predict(Xnew=loc_cv)[0].T

    model_dict[kern_name] = m

    print("%s: CV Error %.4f" %
          (kern_name, np.mean((y_cv - pred_cv[:, k_id])**2)))

np.save("./data/y_tr.npy", y_tr)
np.save("./data/y_cv.npy", y_cv)

np.save("./data/pred_tr.npy", pred_tr)
np.save("./data/pred_cv.npy", pred_cv)

np.save("./data/X_tr.npy", loc_tr)
np.save("./data/X_cv.npy", loc_cv)
pk.dump(model_dict, open("./data/model.pkl", "wb"))

    # # plotting
    # plt.ioff()
    # X = np.linspace(np.min(loc_site[:, 0]), np.max(loc_site[:, 0]), 100)
    # Y = np.linspace(np.min(loc_site[:, 1]), np.max(loc_site[:, 1]), 100)
    # X, Y = np.meshgrid(X, Y)
    # loc_grid = np.vstack((X.flatten(), Y.flatten())).T
    # y_true, _ = generate_pol(x=loc_grid[:, 0], y=loc_grid[:, 1])
    # y_pred = m.predict(Xnew=loc_grid)[0]
    #
    # # plotting prediction
    # Z = y_pred.reshape((100, 100))
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
    #                        linewidth=0.1, antialiased=True)
    # ax.set_zlim(-12, 12)
    # plt.title("%d. %s, cv_error=%.4f" %
    #           (k_id + 1, kern_name, np.mean((y_cv - pred_cv[:, k_id]) ** 2)))
    # plt.savefig("./data/plot/%d_pred_%s.png" % (k_id + 1, kern_name))
    # plt.close()
    #
    # # plotting residual
    # Z = (y_true - y_pred).reshape((100, 100))
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
    #                        linewidth=0.1, antialiased=True)
    # ax.set_zlim(-12, 12)
    # plt.title("%d. %s, cv_error=%.4f" %
    #           (k_id + 1, kern_name, np.mean((y_cv - pred_cv[:, k_id]) ** 2)))
    # plt.savefig("./data/plot/%d_res_%s.png" % (k_id + 1, kern_name))
    # plt.close()
    # plt.ion()


# fit residual process GP
if fit_residual:
    pred_all = np.vstack((pred_tr, pred_cv))
    res = y_obs - pred_all

    pred_eps = np.zeros(shape=(len(y_cv), len(kerns)))
    kernel_eps = gp.kern.RBF(n_dim, ARD=True) + gp.kern.White(n_dim)

    for k_id in range(len(kerns)):
        kern_name = ["Linear", "Poly2", "Poly3", "Poly4",
                     "RBF_ARD", "Matern_12_ARD", "Matern_32_ARD", "Matern_52_ARD",
                     "MLP_ARD", "SpecMix"][k_id]

        y_res = np.atleast_2d(res[:, k_id]).T
        m_eps = gp.models.GPRegression(
            X=loc_site, Y=y_res, kernel=kernel_eps)
        m_eps.optimize(messages=False, max_f_eval=1000)
        pred_eps[:, k_id] = m_eps.predict(Xnew=loc_cv)[0].T
        print("%s: CV Error %.4f" %
              (kern_name, np.mean((y_cv - y_res - pred_eps[:, k_id])**2)))

        # # plotting
        # plt.ioff()
        # X = np.linspace(np.min(loc_site[:, 0]), np.max(loc_site[:, 0]), 100)
        # Y = np.linspace(np.min(loc_site[:, 1]), np.max(loc_site[:, 1]), 100)
        # X, Y = np.meshgrid(X, Y)
        # loc_grid = np.vstack((X.flatten(), Y.flatten())).T
        # y_true, _ = generate_pol(x=loc_grid[:, 0], y=loc_grid[:, 1])
        # eps_pred = m_eps.predict(Xnew=loc_grid)[0]
        #
        # # plotting prediction
        # Z = (pred_eps[:, k_id] - eps_pred).reshape((100, 100))
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
        #                        linewidth=0.1, antialiased=True)
        # ax.set_zlim(-12, 12)
        # plt.title("%d. %s, cv_error=%.4f" %
        #           (k_id + 1, kern_name, np.mean((y_cv - pred_cv[:, k_id]) ** 2)))
        # plt.savefig("./data/plot/%d_eps_%s.png" % (k_id + 1, kern_name))
        # plt.close()
        # plt.ion()


if __name__ == "__main__":

    # visualize location
    sns.regplot(x=loc_site[:, 0], y=loc_site[:, 1], fit_reg=False)

    # visualize pollution surface
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    loc_site_cv = loc_site[cv_id]
    X = np.sort(loc_site_cv[:, 0])
    Y = np.sort(loc_site_cv[:, 1])
    X, Y = np.meshgrid(X, Y)

    # Plot the residual surface.
    X = np.linspace(np.min(loc_site[:, 0]), np.max(loc_site[:, 0]), 100)
    Y = np.linspace(np.min(loc_site[:, 1]), np.max(loc_site[:, 1]), 100)
    X, Y = np.meshgrid(X, Y)
    loc_grid = np.vstack((X.flatten(), Y.flatten())).T
    y_true, _ = generate_pol(x=loc_grid[:, 0], y=loc_grid[:, 1])

    for k_id in range(len(kerns)):
        plt.ioff()
        kern_name = ["Linear", "Poly2", "Poly3", "Poly4",
                     "RBF_ARD", "Matern_12_ARD", "Matern_32_ARD", "Matern_52_ARD",
                     "MLP_ARD", "SpecMix"][k_id]
        pred = model_dict[kern_name].predict(Xnew=loc_grid)[0]

        # plotting prediction
        Z = (y_true - pred).reshape((100, 100))
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0.1, antialiased=True)
        ax.set_zlim(-12, 12)
        plt.title("residual. %s, cv_error=%.4f" %
                  (kern_name, np.mean((y_cv - pred_cv[:, k_id]) ** 2)))
        plt.savefig("./data/plot/resid/%d_resid_%s.png" % (k_id + 1, kern_name))
        plt.close()
        plt.ion()