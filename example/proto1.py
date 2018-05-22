__author__ = "jeremiah"

import os
import datetime
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import tqdm
import pickle as pk

import pandas as pd
import numpy as np
import scipy.interpolate as interpolate

import theano
import theano.tensor as tt
import pymc3 as pm

from aden.model import ensemble_model

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import scipy.linalg as linalg
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from aden.util.misc import array_cosine, spatial_weight_gen, add_dim_name


plot_data = False

# model parameters
linear_spec = True
sparse_weight = True
model_residual = False
link_func = ["logistic", "relu"][0]

name_list = ["Itai", "QD", "Randall"]
data_addr = "./data/proto1/"


################################
# 0. read in base model predictions
################################
# read in data
y_model = []
for name in name_list:
    data_pd = pd.read_csv(data_addr + "%s_2011_align.csv" % name)
    X_model = data_pd[["lon", "lat"]].values.tolist()
    y_model.append(data_pd["pm25"].tolist())

X_model = np.array(X_model)
y_model = np.array(y_model).T

data_id = np.linspace(0, len(X_model)-1, num=10000, dtype=int)
X_model = X_model[data_id]
y_model = y_model[data_id]


if plot_data:
    y = y_model

    cm = plt.cm.get_cmap('jet')
    levels = np.percentile(y.flatten(), np.linspace(0, 100, 101))
    norm = colors.BoundaryNorm(levels, 256)

    plt.scatter(X[:, 0], X[:, 1], c=y[:, 1],
                alpha=0.7, s=0.3,
                cmap=cm, norm=norm)


def simu_proto1(X, pred, n_site=1000, sigma_e=0.1, ls_k=100., alpha=1.,
                add_intercept=True):
    N, K = pred.shape

    # 1. truncate and standardize input
    X_all = X.copy()
    pred_all = pred.copy()

    X_all = (X_all - np.min(X_all, axis=0))/(np.max(X_all, axis=0) - np.min(X_all, axis=0))
    pred_all = (pred_all - np.min(pred_all[:, 0]))/(np.max(pred_all[:, 0]) - np.min(pred_all[:, 0]))
    pred_all = np.clip(pred_all, a_min=0, a_max=1)

    # 2 generate weight
    # model-specific weight
    w_model = np.random.dirichlet(alpha=[alpha] * K)[None, :]
    # location-specific weight
    w_loc = spatial_weight_gen(X=X_all, K=K, ls=ls_k, n_induce=[20, 20])
    # overall weight
    w_all = w_model * w_loc
    w_all = w_all/np.sum(w_all, axis=1)[:, None]

    # 3. generate true surface
    y_true = np.sum(pred_all * w_all, axis=1)[:, None]

    # 4. sample monitor, generate observation
    site_id = np.random.choice(range(N), size=n_site, replace=False)
    w_obs = w_all[site_id]
    X_obs = X_all[site_id]
    y_obs = y_true[site_id] + sigma_e * np.random.normal(size=n_site)[:, None]

    # 5. output corresponding model prediction
    pred_obs = pred_all[site_id]
    if add_intercept:
        intercept_model = np.array([np.mean(y_obs)]*n_site)[:, None]
        pred_obs = np.concatenate((intercept_model, y[site_id]), axis=1)

    return X_obs, y_obs, pred_obs, w_obs, X_all, y_true, pred_all, w_all


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


cv_id = np.linspace(0, len(X_model)-1, num=5000, dtype=int)

model_name = ["Itai", "QD", "Randall"]
f_names = ["f_" + kern_name for kern_name in model_name]


####################################################
# 1. define overall parameter and result container
####################################################
# cv parameter
n_rep = 20
n_fold = 2
kf = KFold(n_splits=n_fold, random_state=100, shuffle=True)

# data parameter
P = X_model.shape[1]
K = len(model_name)

ls_list = [1, 5, 10, 50]
N_list = [5, 15, 25, 50, 75, 100, 150, 200]
sigma_list = [0.1, 0.25, 0.5, 0.75, 1, 1.25]

alpha_w = 1

# model parameter
ls_model = None
temp_prior = 100.

# model container
plot_name = "with_resid" if model_residual else "no_resid"
train_error = np.zeros(shape=(len(N_list), len(sigma_list)))
pred_error = np.zeros(shape=(len(N_list), len(sigma_list)))
oracle_error = np.zeros(shape=(len(N_list), len(sigma_list)))
weight_error = np.zeros(shape=(len(N_list), len(sigma_list)))

for ls_id in range(len(ls_list))[:1]:
    ls_k = ls_list[ls_id]
    print("fixing ls_model to true value....")
    ls_model = ls_k

    for N_id in range(len(N_list))[:1]:
        N = N_list[N_id]

        # define model
        X_obs, y_obs, pred_obs, w_obs, X_all, y_all, pred_all, w_all = \
            simu_proto1(X_model, pred=y_model,
                        n_site=int(N / (1 - 1. / n_fold)), sigma_e=1,
                        ls_k=ls_k, alpha=alpha_w,
                        add_intercept=False)

        y_tr, pred_tr, X_tr = y_obs[:N], pred_obs[:N], X_obs[:N]

        y_tt = theano.shared(y_tr)
        pred_tt = theano.shared(pred_tr)
        X_tt = theano.shared(X_tr)
        ls_tt = theano.shared(ls_model)
        temp_tt = theano.shared(temp_prior)

        model_spec = \
            ensemble_model(y_tt, pred_tt, X_tt=X_tt, ls_tt=ls_tt, temp_tt=temp_tt,
                           K=K, N=N, P=P, model_name=model_name,
                           link_func="logistic", sparse_weight=True,
                           linear_spec=False, eps=1e-12)

        for sigma_id in range(len(sigma_list))[:1]:
            ################################
            # 1. define model and parameters
            ################################
            # define parameters
            sigma_e = sigma_list[sigma_id] * np.std(y_all)

            ################################
            # 3. repeated cv evaluation
            ################################
            train_error_rep = []
            pred_error_rep = []
            oracle_error_rep = []
            weight_error_rep = []

            for rep_id in range(n_rep)[:1]:
                train_error_ls = []
                pred_error_ls = []
                oracle_error_ls = []
                weight_error_ls = []

                # generate data
                X_obs, y_obs, pred_obs, w_obs, \
                X_all, y_all, pred_all, w_all = \
                    simu_proto1(X_model, pred=y_model,
                                n_site=int(N / (1 - 1. / n_fold)),
                                sigma_e=sigma_e,
                                ls_k=ls_k, alpha=alpha_w,
                                add_intercept=False)

                for train_index, test_index in kf.split(X_obs):
                    # prepare train/test batch
                    pred_train, y_train, X_train = \
                        pred_obs[train_index], y_obs[train_index], X_obs[train_index]
                    pred_test, y_test, X_test = \
                        pred_obs[test_index], y_obs[test_index], X_obs[test_index]
                    pred_cv, y_cv, X_cv = \
                        pred_all[cv_id], y_all[cv_id], X_all[cv_id]
                    w_obs_tr, w_obs_tst = w_obs[train_index], w_obs[test_index]

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
                    w_cv = ensemble_pred(X_test, X_train, model_est=model_fit, P=P, ls=ls_model)
                    w_or = ensemble_pred(X_cv, X_train, model_est=model_fit, P=P, ls=ls_model)

                    # training error
                    y_pred_tr = np.sum(pred_train * w_tr, axis=1)[:, None]
                    # cv error
                    y_pred_cv = np.sum(pred_test * w_cv, axis=1)[:, None]
                    # oracle error
                    y_pred_or = np.sum(pred_cv * w_or, axis=1)[:, None]
                    # weight error (cosine loss)
                    w_pred_tr = np.mean(array_cosine(w_tr, w_obs_tr))

                    # model residual process
                    if model_residual:
                        resid_model = XGBRegressor()
                        resid_tr = y_train - y_pred_tr
                        resid_model.fit(X_train, resid_tr)

                        resid_tr = resid_model.predict(X_train)
                        resid_cv = resid_model.predict(X_test)
                        resid_or = resid_model.predict(X_cv)

                        y_pred_tr += resid_tr[:, None]
                        y_pred_cv += resid_cv[:, None]
                        y_pred_or += resid_or[:, None]

                    # record train/test for this batch
                    train_error_ls.append(
                        np.var(y_train - y_pred_tr)/np.var(y_train)
                    )
                    pred_error_ls.append(
                        np.var(y_test - y_pred_cv)/np.var(y_test)
                    )
                    oracle_error_ls.append(
                        np.var(y_cv - y_pred_or)/np.var(y_cv)
                    )
                    weight_error_ls.append(w_pred_tr)

                    print("train:\t %.4f, \t vs.  %.4f, %.4f, %.4f" %
                          (train_error_ls[-1],
                           np.var(y_train - pred_train[:, -3, None]) / np.var(y_train),
                           np.var(y_train - pred_train[:, -2, None]) / np.var(y_train),
                           np.var(y_train - pred_train[:, -1, None]) / np.var(y_train))
                          )
                    print("test:\t %.4f, \t vs.  %.4f, %.4f, %.4f" %
                          (pred_error_ls[-1],
                           np.var(y_test - pred_test[:, -3, None]) / np.var(y_test),
                           np.var(y_test - pred_test[:, -2, None]) / np.var(y_test),
                           np.var(y_test - pred_test[:, -1, None]) / np.var(y_test))
                          )
                    print("valid:\t %.4f, \t vs.  %.4f, %.4f, %.4f" %
                          (oracle_error_ls[-1],
                           np.var(y_cv - pred_cv[:, -3, None]) / np.var(y_cv),
                           np.var(y_cv - pred_cv[:, -2, None]) / np.var(y_cv),
                           np.var(y_cv - pred_cv[:, -1, None]) / np.var(y_cv))
                          )
                    print("weight:\t %.4f: \t  (%.2f, %.2f, %.2f) vs (%.2f, %.2f, %.2f)" %
                          ((weight_error_ls[-1], ) + \
                          tuple(list(np.mean(w_tr, axis=0))) + \
                          tuple(list(np.mean(w_obs_tr, axis=0))))
                          )

                train_error_rep.append(np.mean(np.array(train_error_ls)))
                pred_error_rep.append(np.mean(np.array(pred_error_ls)))
                oracle_error_rep.append(np.mean(np.array(oracle_error_ls)))
                weight_error_rep.append(np.mean(np.array(weight_error_ls)))

            train_error[N_id, sigma_id] = np.median(np.array(train_error_rep))
            pred_error[N_id, sigma_id] = np.median(np.array(pred_error_rep))
            oracle_error[N_id, sigma_id] = np.median(np.array(oracle_error_rep))
            weight_error[N_id, sigma_id] = np.median(np.array(weight_error_rep))

            print("\n\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print(">>>>> result for ls = %d, N = %d, sigma=%.3f done\n\n\n" % (ls_k, N, sigma_e))
            print("train: %.4f, test %.4f, oracle %.4f, weight %.4f" %
                  (train_error[N_id, sigma_id],
                   pred_error[N_id, sigma_id],
                   oracle_error[N_id, sigma_id], weight_error[N_id, sigma_id]))
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n\n")

    np.save("./data/proto1_train_r2_%s_ls%.1f.npy" % (plot_name, ls_k), train_error)
    np.save("./data/proto1_pred_r2_%s_ls%.1f.npy" % (plot_name, ls_k), pred_error)
    np.save("./data/proto1_oracle_r2_%s_ls%.1f.npy" % (plot_name, ls_k), oracle_error)
    np.save("./data/proto1_weight_cos_%s_ls%.1f.npy" % (plot_name, ls_k), weight_error)

    # save result
    # train_error = np.load("./data/proto1_train_error_no_resid.npy")
    # pred_error = np.load("./data/proto1_pred_error_no_resid.npy")
    # oracle_error = np.load("./data/proto1_oracle_error_no_resid.npy")
    # weight_error = np.load("./data/proto1_weight_error_no_resid.npy")


    np.savetxt("./data/proto1_train_r2_ls%.1f.txt" % ls_k,
               add_dim_name(1-train_error, row_name=N_list, col_name=sigma_list),
               delimiter=' & ', fmt='%.4f', newline=' \\\\\n')
    np.savetxt("./data/proto1_pred_r2_ls%.1f.txt" % ls_k,
               add_dim_name(1-pred_error, row_name=N_list, col_name=sigma_list),
               delimiter=' & ', fmt='%.4f', newline=' \\\\\n')
    np.savetxt("./data/proto1_oracle_r2_ls%.1f.txt" % ls_k,
               add_dim_name(1-oracle_error, row_name=N_list, col_name=sigma_list),
               delimiter=' & ', fmt='%.4f', newline=' \\\\\n')
    np.savetxt("./data/proto1_weight_cos_ls%.1f.txt" % ls_k,
               add_dim_name(weight_error, row_name=N_list, col_name=sigma_list),
               delimiter=' & ', fmt='%.4f', newline=' \\\\\n')