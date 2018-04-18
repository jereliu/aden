__author__ = "jeremiah"

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import theano
import theano.tensor as tt
import numpy as np

import pickle as pk

# read in base model predictions
y = np.load("./data/y_tr.npy")
X = np.load("./data/X_tr.npy")

pred = np.load("./data/pred_tr.npy")

model_name = ["Linear", "Poly2", "Poly3", "Poly4",
              "RBF_ARD", "Matern_12_ARD", "Matern_32_ARD", "Matern_52_ARD",
              "MLP_ARD", "SpecMix"]
parametrization = ["dirichlet", "logistic"][0]

# build prediction location
loc_site_cv = np.load("./data/X_cv.npy")
loc_X = np.linspace(np.min(loc_site_cv[:, 0]), np.max(loc_site_cv[:, 0]), 20)
loc_Y = np.linspace(np.min(loc_site_cv[:, 1]), np.max(loc_site_cv[:, 1]), 20)
loc_X, loc_Y = np.array(np.meshgrid(loc_X, loc_Y))
X_pred = np.array([loc_X.flatten(), loc_Y.flatten()]).T


# read in base model predictions
import pymc3 as pm

N, P = X.shape
N, K = pred.shape
ls = 2

pred_tt = theano.shared(pred)
y_tt = theano.shared(y)

for parametrization in ["dirichlet", "logistic"][1:]:
    with pm.Model() as ensemble_model:
        # define model-specific gp
        cov_func = pm.gp.cov.ExpQuad(input_dim=P, ls=ls)
        gp = pm.gp.Latent(cov_func=cov_func)
        f = []
        for k in range(K):
            f.append(gp.prior("f_"+model_name[k], X=X))
        f = tt.stack(f)

        # transform into Dirichlet ensemble
        if parametrization == "logistic":
            a = f.T
            w = pm.Deterministic("w", tt.nnet.softmax(a))
        elif parametrization == "dirichlet":
            # gamma construction
            a_0 = pm.Gamma(name="a_0", alpha=pm.math.exp(f.T), beta=1, shape=(N, K))
            a = a_0 / a_0.norm(1, axis=1).reshape((a_0.shape[0], 1))
            w = pm.Deterministic("w", a)
        else:
            raise ValueError("parametrization %s not supported" % parametrization)

        # connect with outcome
        # sigma = pm.InverseGamma(alpha=1, beta=1)
        sigma = 1
        mu = pm.Deterministic("pred", (w*pred_tt).sum(axis=1))
        pm.Normal("obs", mu=mu, sd=sigma, observed=y_tt.T)

    # model_map = pm.find_MAP(model=ensemble_model)


    with ensemble_model:
        # draw 10000 posterior samples
        mc_step = pm.NUTS()
        trace = pm.sample(draws=10000, tune=10000, step=mc_step,
                          nchains=1, cores=1)

    pk.dump(trace,
            open("./data/ensemble_" + parametrization + ".pkl", "wb"))

