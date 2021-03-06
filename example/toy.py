__author__ = "jeremiah"

import os
import datetime
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import theano
import theano.tensor as tt
import numpy as np

import pickle as pk

from aden.util.sp_process_gen import generate_data
from aden.model import ensemble_model

run_nuts = False

linear_spec = True
sparse_weight = True
link_func = ["dirichlet", "logistic", "relu"][1]

################################
# 0. read in base model predictions
################################

y = np.load("./data/y_tr.npy")
X = np.load("./data/X_tr.npy")

pred = np.load("./data/pred_tr.npy")

model_name = ["Intercept", "Linear", "Poly2", "Poly3", "Poly4",
              "RBF_ARD", "Matern_12_ARD", "Matern_32_ARD", "Matern_52_ARD",
              "MLP_ARD", "SpecMix"]

# build prediction location
loc_site_cv = np.load("./data/X_cv.npy")

n_test_point = 100
loc_X = np.linspace(np.min(loc_site_cv[:, 0]), np.max(loc_site_cv[:, 0]), n_test_point)
loc_Y = np.linspace(np.min(loc_site_cv[:, 1]), np.max(loc_site_cv[:, 1]), n_test_point)
loc_X, loc_Y = np.array(np.meshgrid(loc_X, loc_Y))
X_pred = np.array([loc_X.flatten(), loc_Y.flatten()]).T

# build cv prediction
model_dict = pk.load(open("./data/model.pkl", "rb"))

y_cv, _ = generate_data(x=X_pred[:, 0], y=X_pred[:, 1])
pred_cv = [model_dict[k_name].predict(Xnew=X_pred)[0]
           for k_name in model_name]
pred_cv = np.array(pred_cv)[:, :, 0].T

cv_error = np.mean((y_cv - pred_cv)**2, 0)

################################
# 1. define ensemble model
################################
import pymc3 as pm

n_fold = 5
train_size = int(250 * (1 - 1./n_fold) if n_fold > 1 else 250)

y_train, pred_train, X_train = \
    y[:train_size], pred[:train_size], X[:train_size]

N, P = X_train.shape
N, K = pred_train.shape
ls = 2.
temp = 2.

y_tt = theano.shared(y_train)
pred_tt = theano.shared(pred_train)
X_tt = theano.shared(X_train)
ls_tt = theano.shared(ls)
temp_tt = theano.shared(temp)

model_spec = \
    ensemble_model(y_tt, pred_tt, X_tt=X_tt, ls_tt=ls_tt, temp_tt=0.5,
                   K=K, N=N, P=P, model_name=model_name,
                   link_func="logistic", sparse_weight=True,
                   linear_spec=False, eps=1e-12)

if run_nuts:
    with model_spec:
        # draw 10000 posterior samples
        mc_step = pm.NUTS()
        trace = pm.sample(draws=5000, tune=5000, step=mc_step,
                          nchains=1, cores=1)

    pk.dump(trace,
            open("./data/ensemble_" + link_func + ".pkl", "wb"))


#####################################
# 2. define residual process model
#####################################
from xgboost import XGBRegressor

# # produce residual
# resid = y.flatten() - pred[:, 0]
#
# # fit
# resid_model = XGBRegressor()
# resid_model.fit(X, resid)
#
# trees = resid_model.apply(X)


################################
# 3. cv evaluation
################################
import scipy.linalg as linalg
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

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
model_residual = False
kf = KFold(n_splits=n_fold, random_state=100, shuffle=True)

ls_list = np.linspace(2., 2.5, 5)

for model_residual in [False]:
    plot_name = "with_resid" if model_residual else "no_resid"
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
            # with model_spec:
            #     model_fit = pm.fit(n=100000, method=pm.ADVI())
            #     trace = model_fit.sample(1000)
            model_fit = pm.find_MAP(model=model_spec)

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

            # model residual process
            if model_residual:
                resid_model = XGBRegressor()
                resid_tr = y_train - y_pred_tr
                resid_model.fit(X_train, resid_tr)

                resid_tr = resid_model.predict(X_train)
                resid_cv = resid_model.predict(X_test)
                resid_or = resid_model.predict(X_pred)

                y_pred_tr += resid_tr[:, None]
                y_pred_cv += resid_cv[:, None]
                y_pred_or += resid_or[:, None]

            # record train/test for this batch
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

        np.save("./data/train_error_%s.npy" % plot_name, train_error)
        np.save("./data/pred_error_%s.npy" % plot_name, pred_error)
        np.save("./data/oracle_error_%s.npy" % plot_name, oracle_error)

# plot result

import matplotlib.pyplot as plt
import seaborn as sns

plot_name = "with_resid"
train_error_resid = np.load("./data/train_error_%s.npy" % plot_name)
pred_error_resid = np.load("./data/pred_error_%s.npy" % plot_name)
oracle_error_resid = np.load("./data/oracle_error_%s.npy" % plot_name)

plot_name = "no_resid"
train_error_noresid = np.load("./data/train_error_%s.npy" % plot_name)
pred_error_noresid = np.load("./data/pred_error_%s.npy" % plot_name)
oracle_error_noresid = np.load("./data/oracle_error_%s.npy" % plot_name)

sns.set_style("darkgrid")

plt.figure(figsize=(8, 4))
plt.plot(ls_list, train_error_noresid, label="base model ensemble only")
plt.plot(ls_list, train_error_resid, label="base model ensemble + residual process")
plt.legend(loc=1, borderaxespad=0.)
plt.title("training error, based on y_obs")
plt.savefig("./report/np_int/plot/cv_error_obs_resid.png")

plt.figure(figsize=(8, 4))
plt.plot(ls_list, pred_error_noresid, label="base model ensemble only")
plt.plot(ls_list, pred_error_resid, label="base model ensemble + residual process")
plt.legend(loc=1, borderaxespad=0.)
plt.title("5-fold cv error, based on y_obs")
plt.savefig("./report/np_int/plot/cv_error_cv_resid.png")

plt.figure(figsize=(8, 4))
plt.plot(ls_list, oracle_error_noresid, label="base model ensemble only")
plt.plot(ls_list, oracle_error_resid, label="base model ensemble + residual process")
plt.legend(loc=1, borderaxespad=0.)
plt.title("5-fold cv error, based on y_new")
plt.savefig("./report/np_int/plot/cv_error_pred_resid.png")
