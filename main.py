__author__ = "jeremiah"

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import theano
import theano.tensor as tt
import numpy as np

import pickle as pk

from aden.util.sp_process_gen import generate_pol

run_nuts = False
parametrization = "logistic"

################################
# 0. read in base model predictions
################################
y = np.load("./data/y_tr.npy")
X = np.load("./data/X_tr.npy")

pred = np.load("./data/pred_tr.npy")

model_name = ["Linear", "Poly2", "Poly3", "Poly4",
              "RBF_ARD", "Matern_12_ARD", "Matern_32_ARD", "Matern_52_ARD",
              "MLP_ARD", "SpecMix"]
parametrization = ["dirichlet", "logistic"][1]

# build prediction location
loc_site_cv = np.load("./data/X_cv.npy")

n_test_point = 100
loc_X = np.linspace(np.min(loc_site_cv[:, 0]), np.max(loc_site_cv[:, 0]), n_test_point)
loc_Y = np.linspace(np.min(loc_site_cv[:, 1]), np.max(loc_site_cv[:, 1]), n_test_point)
loc_X, loc_Y = np.array(np.meshgrid(loc_X, loc_Y))
X_pred = np.array([loc_X.flatten(), loc_Y.flatten()]).T

# build cv prediction
model_dict = pk.load(open("./data/model.pkl", "rb"))

y_cv, _ = generate_pol(x=X_pred[:, 0], y=X_pred[:, 1])
pred_cv = [model_dict[k_name].predict(Xnew=X_pred)[0]
           for k_name in model_name]
pred_cv = np.array(pred_cv)[:, :, 0].T

cv_error = np.mean((y_cv - pred_cv)**2, 0)

################################
# 1. define model, run sample
################################
import pymc3 as pm

n_fold = 10
train_size = int(250 * (1 - 1./n_fold) if n_fold > 1 else 250)

y_train, pred_train, X_train = \
    y[:train_size], pred[:train_size], X[:train_size]

N, P = X_train.shape
N, K = pred_train.shape
ls = 2.

y_tt = theano.shared(y_train)
pred_tt = theano.shared(pred_train)
X_tt = theano.shared(X_train)
ls_tt = theano.shared(ls)

with pm.Model() as ensemble_model:
    # define model-specific gp
    # change ls to ls_tt, or vice versa
    cov_func = pm.gp.cov.ExpQuad(input_dim=P, ls=ls_tt)
    gp = pm.gp.Latent(cov_func=cov_func)
    f = []
    for k in range(K):
        # change X to X_tt, or vice versa
        f.append(gp.prior("f_"+model_name[k], X=X_tt, shape=N))
    f = tt.stack(f)

    # transform into Dirichlet ensemble
    if parametrization == "logistic":
        a = f.T
        w = pm.Deterministic("w", tt.nnet.softmax(a))
    elif parametrization == "dirichlet":
        # gamma construction
        a_0 = pm.Gamma(name="a_0", alpha=pm.math.exp(f.T),
                       beta=1, shape=(N, K))
        a = a_0 / a_0.norm(1, axis=1).reshape((a_0.shape[0], 1))
        w = pm.Deterministic("w", a)
    else:
        raise ValueError("parametrization %s not supported" %
                         parametrization)

    # connect with outcome
    # sigma = pm.InverseGamma(alpha=1, beta=1)
    sigma = 1
    mu = pm.Deterministic("pred", (w*pred_tt).sum(axis=1))
    pm.Normal("obs", mu=mu, sd=sigma, observed=y_tt.T)


if run_nuts:
    with ensemble_model:
        # draw 10000 posterior samples
        mc_step = pm.NUTS()
        trace = pm.sample(draws=10000, tune=10000, step=mc_step,
                          nchains=1, cores=1)

    pk.dump(trace,
            open("./data/ensemble_" + parametrization + ".pkl", "wb"))


################################
# 2. cv evaluation
################################
import scipy.linalg as linalg
from sklearn.model_selection import KFold

f_names = ["f_" + kern_name for kern_name in model_name]


def ensemble_pred(X_test, X_train, model_est, P, ls):
    K_pred = pm.gp.cov.ExpQuad(input_dim=P, ls=ls).full(X_test, X_train).eval()
    K_train = pm.gp.cov.ExpQuad(input_dim=P, ls=ls).full(X_train, X_train).eval()
    K_train_inv = linalg.pinv(K_train)
    K_pre = K_pred.dot(K_train_inv)

    # f_pred_list = [K_pre.dot(np.mean(model_est[f_name], axis=0))
    #                for f_name in f_names]
    f_pred_list = [K_pre.dot(model_est[f_name]) for f_name in f_names]
    f_pred_list = np.array(f_pred_list)
    w_pred = np.exp(f_pred_list - np.max(f_pred_list, axis=0))
    w_pred = w_pred/np.sum(w_pred, axis=0)

    return w_pred.T


# 2.2 run cv


kf = KFold(n_splits=n_fold, random_state=100, shuffle=True)

ls_list = np.linspace(1.5, 3., 20)
train_error = []
pred_error = []
oracle_error = []

for ls in ls_list:
    ls_tt.set_value(ls)
    train_error_ls = []
    pred_error_ls = []
    oracle_error_ls = []

    for train_index, test_index in kf.split(X):
        # prepare train/test batch
        pred_train, y_train, X_train = pred[train_index], y[train_index], X[train_index]
        pred_test, y_test, X_test = pred[test_index], y[test_index], X[test_index]

        pred_tt.set_value(pred_train)
        y_tt.set_value(y_train)
        X_tt.set_value(X_train)

        # fit model
        # with ensemble_model:
        #     model_fit = pm.fit(n=100000, method=pm.ADVI())
        #     trace = model_fit.sample(1000)
        model_fit = pm.find_MAP(model=ensemble_model)

        # do prediction
        w_tr = model_fit["w"] # np.mean(trace["w"], axis=0)
        w_cv = ensemble_pred(X_test, X_train, model_est=model_fit, P=P, ls=ls)
        w_or = ensemble_pred(X_pred, X_train, model_est=model_fit, P=P, ls=ls)

        # training error
        y_pred_tr = np.sum(pred_train * w_tr, axis=1)[:, None]
        # cv error
        y_pred_cv = np.sum(pred_test * w_cv, axis=1)[:, None]
        # oracle error
        y_pred_or = np.sum(pred_cv * w_or, axis=1)[:, None]

        train_error_ls.append(np.mean((y_train - y_pred_tr)**2))
        pred_error_ls.append(np.mean((y_test - y_pred_cv)**2))
        oracle_error_ls.append(np.mean((y_cv - y_pred_or)**2))

        print("train: %.4f, test %.4f, oracle %.4f" %
              (train_error_ls[-1], pred_error_ls[-1], oracle_error_ls[-1]))

    train_error.append(np.mean(np.array(train_error_ls)))
    pred_error.append(np.mean(np.array(pred_error_ls)))
    oracle_error.append(np.mean(np.array(oracle_error_ls)))

    print("\n\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(">>>>> result for ls = %.3f done\n\n\n" % ls)
    print("train: %.4f, test %.4f, oracle %.4f" %
          (train_error[-1], pred_error[-1], oracle_error[-1]))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n\n")

# plot result
import matplotlib.pyplot as plt

plt.plot(ls_list, train_error, label="training error, based on y_obs")
plt.plot(ls_list, pred_error, label="5-fold cv error, based on y_obs")
plt.legend(loc=1, borderaxespad=0.)

plt.plot(ls_list, oracle_error, label="5-fold cv error, based on y_pred")
plt.legend(loc=1, borderaxespad=0.)
