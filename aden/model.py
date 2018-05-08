__author__ = "jeremiah"

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import pymc3 as pm
import theano.tensor as tt


class aden(object):
    def __init__(self, model_dict, parametrization="logistic"):
        self.model = model_dict

        with pm.Model() as ensemble_model:
            # define model-specific gp
            cov_func = pm.gp.cov.ExpQuad(input_dim=P, ls=ls)
            gp = pm.gp.Latent(cov_func=cov_func)
            f = []
            for k in range(K):
                f.append(gp.prior("f_" + model_name[k], X=X))
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
            mu = pm.Deterministic("pred", (w * pred_tt).sum(axis=1))
            pm.Normal("obs", mu=mu, sd=sigma, observed=y_tt.T)

        self.model = ensemble_model