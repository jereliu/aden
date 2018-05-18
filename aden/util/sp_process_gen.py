__author__ = ["jeremiah"]

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import pandas as pd
import pickle as pk
import random

import numpy as np
import seaborn as sns

import GPy as gp
from aden.kernel.spectral_mixture import SpectralMixture

type = ["toy", "simu"][1]
noise = 0.01

if type == "toy":
    # random sample location
    n_site = 500
    n_dim = 2
    np.random.seed(100)
    loc_site = np.random.normal(size=(n_site, n_dim))
elif type == "simu":
    data = pd.read_csv("./data/simu/Itai_2011_subsample.csv")
    n_site = len(data)
    n_dim = 2
    loc_site = np.array(data[["lon", "lat"]].values.tolist())


# random sample response
def generate_data(x, y, noise=0.1, type="simu"):
    if type == "toy":
        y_true = 0.2 * x + 0.5 * y + np.sqrt(x**2 + y**2 + 5* np.cos(x*y)) + \
                 np.sin(x) + np.cos(y) + np.logaddexp(x*y, x)
        y_true = np.atleast_2d(y_true).T
    elif type == "simu":
        print("reading data and standardize to range [0, 1]..")
        data = pd.read_csv("./data/simu/Itai_2011_subsample.csv")
        y_true = np.atleast_2d(data["pm25"].tolist()).T
        y_true = (y_true - np.min(y_true))/(np.max(y_true) - np.min(y_true))

    y_obs = y_true + np.random.normal(scale=noise, size=y_true.shape)
    return y_obs, y_true


if __name__ == "__main__":
    fit_residual = False

    y_obs, y_true = generate_data(x=loc_site[:, 0],
                                  y=loc_site[:, 1],
                                  noise=noise,
                                  type=type)


    # generate base prediction
    kerns = [gp.kern.Bias(n_dim), # + gp.kern.White(n_dim),
             gp.kern.Linear(n_dim), # + gp.kern.White(n_dim),
             #gp.kern.Poly(n_dim, order=2), # + gp.kern.White(n_dim),
             #gp.kern.Poly(n_dim, order=3), # + gp.kern.White(n_dim),
             #gp.kern.Poly(n_dim, order=4), # + gp.kern.White(n_dim),
             gp.kern.RBF(n_dim, ARD=True), # + gp.kern.White(n_dim),
             gp.kern.OU(n_dim, ARD=True),  # + gp.kern.White(n_dim),
             gp.kern.Matern32(n_dim, ARD=True),  # + gp.kern.White(n_dim),
             gp.kern.Matern52(n_dim, ARD=True), # + gp.kern.White(n_dim),
             gp.kern.MLP(n_dim, ARD=True), # + gp.kern.White(n_dim),
             SpectralMixture(n_dim)]# + gp.kern.White(n_dim)]

    # create simple GP model
    full_id = range(len(data))
    tr_id = np.random.choice(full_id, 5000, replace=False)
    cv_id = list(set(full_id) - set(tr_id))
    # cv_id = np.random.choice(cv_id, 5000, replace=False)

    loc_tr, loc_cv = loc_site[tr_id], loc_site[cv_id]
    y_tr, y_cv = y_obs[tr_id], y_obs[cv_id]

    pred_tr = np.zeros(shape=(len(y_tr), len(kerns)))
    pred_cv = np.zeros(shape=(len(y_cv), len(kerns)))

    model_dict = dict()

    for k_id in range(len(kerns)):
        kern_name = ["Intercept",
                     "Linear",
                     #"Poly2", "Poly3", "Poly4",
                     "RBF_ARD",
                     "Matern_12_ARD", "Matern_32_ARD", "Matern_52_ARD",
                     "MLP_ARD", "SpecMix"][k_id]

        kernel = kerns[k_id]

        print("\n==========================")
        print("Building model '%s'" % kern_name)
        m = gp.models.SparseGPRegression(
            X=loc_tr, Y=y_tr, kernel=kernel, num_inducing=500)
        print("Optimizing..")
        m.optimize(messages=True, max_f_eval=1000)
        print("Predicting..")
        pred_tr[:, k_id] = m.predict(Xnew=loc_tr)[0].T
        pred_cv[:, k_id] = m.predict(Xnew=loc_cv)[0].T

        model_dict[kern_name] = m

        print("%s: CV Error %.4f" %
              (kern_name, np.mean((y_cv - pred_cv[:, k_id])**2)))

    np.save("./data/%s/y_tr.npy" % type, y_tr)
    np.save("./data/%s/y_cv.npy" % type, y_cv)

    np.save("./data/%s/pred_tr.npy" % type, pred_tr)
    np.save("./data/%s/pred_cv.npy" % type, pred_cv)

    np.save("./data/%s/X_tr.npy" % type, loc_tr)
    np.save("./data/%s/X_cv.npy" % type, loc_cv)
    pk.dump(model_dict, open("./data/%s/model.pkl" % type, "wb"))

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
            kern_name = ["Linear",
                         #"Poly2", "Poly3", "Poly4",
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



    # visualize location
    sns.regplot(x=loc_tr[:, 0], y=loc_tr[:, 1], fit_reg=False)

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
    y_true, _ = generate_data(x=loc_grid[:, 0], y=loc_grid[:, 1])

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