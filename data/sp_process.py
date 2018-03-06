import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import numpy as np
import seaborn as sns

import GPy as gp

# random sample location
n_site = 200
n_dim = 2
loc_site = np.random.normal(size=(n_site, n_dim))
noise = 0.1


# random sample response
def generate_pol(x, y, noise=0.1):
    y_true = 0.2 * x + 0.5 * y + np.sqrt(x**2 + y**2) + np.sin(x) + np.cos(x)
    y_true = np.atleast_2d(y_true).T

    y_obs = y_true + np.random.normal(scale=noise, size=y_true.shape)
    return y_obs, y_true


y_obs, y_true = generate_pol(x=loc_site[:, 0], y=loc_site[:, 1], noise=noise)


# generate base prediction
kerns = [gp.kern.Linear(n_dim) + gp.kern.White(n_dim),
         gp.kern.Poly(n_dim, order=3) + gp.kern.White(n_dim),
         gp.kern.RBF(n_dim, ARD=True) + gp.kern.White(n_dim),
         gp.kern.Matern52(n_dim, ARD=True) + gp.kern.White(n_dim),
         gp.kern.MLP(n_dim, ARD=True) + gp.kern.White(n_dim)]

# create simple GP model
tr_id = range(n_site/2)
cv_id = range(n_site/2, n_site)

loc_tr, loc_cv = loc_site[tr_id], loc_site[cv_id]
y_tr, y_cv = y_obs[tr_id], y_obs[cv_id]

pred_tr = np.zeros(shape=(len(y_tr), len(kerns)))
pred_cv = np.zeros(shape=(len(y_cv), len(kerns)))

for k_id in range(len(kerns)):
    kernel = kerns[k_id]
    m = gp.models.GPRegression(X=loc_tr, Y=y_tr, kernel=kernel)
    m.optimize(messages=False, max_f_eval=1000)
    pred_tr[:, k_id] = m.predict(Xnew=loc_tr)[0].T
    pred_cv[:, k_id] = m.predict(Xnew=loc_cv)[0].T
    print(np.mean((y_tr - pred_tr[:, k_id])**2)/np.var(y_tr))
    print(np.mean((y_cv - pred_cv[:, k_id])**2)/np.var(y_cv))

np.save("./data/pred", pred_tr)

if __name__ == "__main__":

    # visualize location
    sns.regplot(x=loc_site[:, 0], y=loc_site[:, 1], fit_reg=False)

    # visualize pollution surface
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X = np.sort(loc_site[:, 0])
    Y = np.sort(loc_site[:, 1])
    X, Y = np.meshgrid(X, Y)
    R_obs, R_true = generate_pol(X, Y, noise=0.1)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, R_true, cmap=cm.coolwarm,
                           linewidth=0.1, antialiased=True)
