__author__ = "jeremiah"

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import pickle as pk
import pandas as pd

import numpy as np
import numpy.linalg as linalg

import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import seaborn as sns

trace = pk.load(open("./data/ensemble_logistic.pkl", "rb"))

# trace = pk.load(open("./data/ensemble_dirichlet.pkl", "rb"))


model_name = ["Linear", "Poly2", "Poly3", "Poly4",
              "RBF_ARD", "Matern_12_ARD", "Matern_32_ARD", "Matern_52_ARD",
              "MLP_ARD", "SpecMix"]


#######################
# 1. weights
#######################

# violin plot
w_sample = trace["w"][8000:]

n_sample, N, K = w_sample.shape
data = w_sample.reshape([n_sample * N, K]).swapaxes(0, 1).flatten()
pos = np.repeat(model_name, n_sample * N)

unit_weight_data = \
    pd.DataFrame({"unit_id": pos.astype('str'), "weight value": data})

ax = sns.violinplot(x="unit_id", y="weight value",
                    scale="area", data=unit_weight_data, whis=1)

#######################
# 2. pollution surface
#######################
import pymc3 as pm

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

n_test_point = 100

X_tr = np.load("./data/X_tr.npy")
loc_site_cv = np.load("./data/X_cv.npy")
loc_X = np.linspace(np.min(loc_site_cv[:, 0]), np.max(loc_site_cv[:, 0]), n_test_point)
loc_Y = np.linspace(np.min(loc_site_cv[:, 1]), np.max(loc_site_cv[:, 1]), n_test_point)
X, Y = np.meshgrid(loc_X, loc_Y)
X_pred = np.c_[np.ravel(X), np.ravel(Y)]

N, P = X_tr.shape

#######################
# 2.0. estimate parameters for GP posterior predictive
ls = 2

K_pred = pm.gp.cov.ExpQuad(input_dim=P, ls=ls).full(X_pred, X_tr).eval()
K_train = pm.gp.cov.ExpQuad(input_dim=P, ls=ls).full(X_tr, X_tr).eval()
K_train_inv = linalg.pinv(K_train)

# produce matrix to pre-multiply observation
K_pre = K_pred.dot(K_train_inv)

f_pred_list = []
for k_id in range(len(model_name)):
    kern_name = model_name[k_id]
    f_obs_mean = np.mean(trace["f_" + kern_name], axis=0)
    f_pred = K_pre.dot(f_obs_mean)
    f_pred_list.append(f_pred)

f_pred_list = np.array(f_pred_list)
w_pred_list = np.exp(f_pred_list)
w_pred_list = w_pred_list/np.sum(w_pred_list, axis=0)


# produce matrix characterizing uncertainty (covariance for GP posterior)
K_test = pm.gp.cov.ExpQuad(input_dim=P, ls=ls).full(X_pred, X_pred).eval()
K_pred = pm.gp.cov.ExpQuad(input_dim=P, ls=ls).full(X_pred, X_tr).eval()

K_unc = K_test - K_pred.dot(K_train_inv).dot(K_pred.T)

#######################
# 2.1 surface for ensemble weight


plt.ioff()
for k_id in range(len(model_name)):
    kern_name = model_name[k_id]
    #
    Z = w_pred_list[k_id].reshape(X.shape)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0.1, antialiased=True)
    ax.set_zlim(0, 0.2)
    plt.title("%d. %s, mean weight=%.4f" %
              (k_id + 1, kern_name, np.mean(Z)))
    plt.savefig("./data/plot/weight/%d_weight_%s.png" % (k_id + 1, kern_name))
    plt.close()

plt.ion()

#######################
# 2.2 surface for ensemble uncertainty
# TODO:

plt.ioff()

Z = np.diag(K_unc).reshape(X.shape)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0.1, antialiased=True)

plt.title("posterior predictive variance")
plt.savefig("./data/plot/weight/posterior_var.png")
plt.close()

plt.ion()