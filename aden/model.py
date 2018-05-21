__author__ = ["jeremiah"]

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import numpy as np
import scipy.linalg as linalg

import pymc3 as pm
import theano.tensor as tt


def ensemble_model(y_tt, pred_tt, X_tt, ls_tt, temp_tt,
                   K, N, P, model_name=None,
                   link_func="logistic", sparse_weight=True,
                   linear_spec=False, eps=1e-12):
    """

    :param y_tt: observation
    :param pred_tt: prediction from base models
    :param X_tt: input features
    :param ls_tt: length scale for variance
    :param temp_tt: prior mean for temperature
    :param K: num of base models
    :param N: num of observations
    :param P: num of features
    :param model_name: list of names for base models
    :param link_func: type of link function
    :param sparse_weight: whether to model temperature for logistic link function
    :param linear_spec: use linear specification to accelerate
    :param eps:
    :return:
    """
    if model_name is None:
        model_name = np.arange(K)
    if linear_spec:
        K_train = pm.gp.cov.ExpQuad(input_dim=P, ls=ls_tt).full(X_tt, X_tt).eval()
        U_train, s, _ = linalg.svd(K_train)
        idx_train = np.where(s > eps)[0]
        U_train = np.dot(U_train[:, idx_train], np.diag(np.sqrt(s[idx_train])))
        Np = len(idx_train)

    with pm.Model() as final_model:
        # define model-specific gp
        # ls = pm.InverseGamma('ls', alpha=2, beta=1)
        # change ls to ls_tt, or vice versa

        if linear_spec:
            beta = []
            f = []
            for k in range(K):
                # change X to X_tt, or vice versa
                beta.append(
                    pm.Normal("b_"+str(model_name[k]), mu=0, sd=1, shape=Np)
                )
                f.append(
                    pm.Deterministic(
                        "f_"+str(model_name[k]), tt.dot(U_train, beta[k])))
            beta = tt.stack(beta)
            f = tt.stack(f)

        else:
            cov_func = pm.gp.cov.ExpQuad(input_dim=P, ls=ls_tt)
            gp = pm.gp.Latent(cov_func=cov_func)
            f = []
            for k in range(K):
                # change X to X_tt, or vice versa
                f.append(gp.prior("f_"+model_name[k], X=X_tt, shape=N))
            f = tt.stack(f)

        # transform into Dirichlet ensemble
        if link_func == "logistic":
            if sparse_weight:
                temp = pm.Gamma('T', mu=temp_tt, sd=10)
            else:
                temp = temp_tt
            a = f.T * temp
            w = pm.Deterministic("w", tt.nnet.softmax(a))
        elif link_func == "relu":
            # relu construction
            a = tt.nnet.relu(f.T)
            a = a / a.norm(1, axis=1).reshape((a.shape[0], 1))
            w = pm.Deterministic("w", a)
        elif link_func == "dirichlet":
            # gamma construction
            a_0 = pm.Gamma(name="a_0", alpha=pm.math.exp(f.T),
                           beta=1, shape=(N, K))
            a = a_0 / a_0.norm(1, axis=1).reshape((a_0.shape[0], 1))
            w = pm.Deterministic("w", a)
        else:
            raise ValueError("link function %s not supported" %
                             link_func)

        # connect with outcome
        # sigma = pm.InverseGamma(alpha=1, beta=1)
        sigma = 1
        mu = pm.Deterministic("pred", (w*pred_tt).sum(axis=1))
        pm.Normal("obs", mu=mu, sd=sigma, observed=y_tt.T)

    return final_model


# class aden(object):
#     def __init__(self, model_dict, parametrization="logistic"):
#         self.model = model_dict
#
#         with pm.Model() as ensemble_model:
#             # define model-specific gp
#             cov_func = pm.gp.cov.ExpQuad(input_dim=P, ls=ls)
#             gp = pm.gp.Latent(cov_func=cov_func)
#             f = []
#             for k in range(K):
#                 f.append(gp.prior("f_" + model_name[k], X=X))
#             f = tt.stack(f)
#
#             # transform into Dirichlet ensemble
#             if parametrization == "logistic":
#                 a = f.T
#                 w = pm.Deterministic("w", tt.nnet.softmax(a))
#             elif parametrization == "dirichlet":
#                 # gamma construction
#                 a_0 = pm.Gamma(name="a_0", alpha=pm.math.exp(f.T), beta=1, shape=(N, K))
#                 a = a_0 / a_0.norm(1, axis=1).reshape((a_0.shape[0], 1))
#                 w = pm.Deterministic("w", a)
#             else:
#                 raise ValueError("parametrization %s not supported" % parametrization)
#
#             # connect with outcome
#             # sigma = pm.InverseGamma(alpha=1, beta=1)
#             sigma = 1
#             mu = pm.Deterministic("pred", (w * pred_tt).sum(axis=1))
#             pm.Normal("obs", mu=mu, sd=sigma, observed=y_tt.T)
#
#         self.model = ensemble_model