import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf
import scipy.spatial.distance

from sde.sde_learning_network import \
(
    SDEIntegrators
)


def sample_data(drift_diffusivity, step_size, n_dimensions, low, high, n_pts, rng, n_subsample=10):
    x_data = rng.uniform(low=low, high=high, size=(n_pts, n_dimensions))
    y_data = x_data.copy()
    for k in range(n_subsample):
        y_data = np.row_stack([
            SDEIntegrators.euler_maruyama(y_data[k, :],
                                          step_size / n_subsample,
                                          drift_diffusivity,
                                          rng)
            for k in range(x_data.shape[0])
        ])

    return x_data, y_data


def plot_results_functions(apx_drift_diffusivity, true_drift_diffusivity,
                           x_data, y_data, rng,
                           data_transform_network=None,
                           data_transform_true=None):
    if data_transform_network is None:
        def data_transform_network(x):
            return x
    if data_transform_true is None:
        def data_transform_true(x):
            return x

    mean_network, std_network = apx_drift_diffusivity(data_transform_network(x_data).astype(np.float32))

    mean_network = keras.backend.eval(mean_network)
    std_network = keras.backend.eval(std_network)

    if std_network.shape != mean_network.shape:
        std_network = keras.backend.eval(tf.linalg.diag_part(std_network))

    n_dimensions = x_data.shape[1]

    ms = 0.25  # marker size

    x_data_transformed = data_transform_true(x_data)
    true_drift_evaluated, true_std_evaluated = true_drift_diffusivity(x_data_transformed)

    if true_std_evaluated.shape != true_drift_evaluated.shape:
        true_std_evaluated = tf.linalg.diag_part(true_std_evaluated)

    if n_dimensions == 1:
        mean_network = mean_network.reshape(-1, 1)
        std_network = std_network.reshape(-1, 1)
        
        idx_ = np.argsort(x_data_transformed.ravel())

        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax[0].plot(x_data_transformed[idx_], true_drift_evaluated[idx_], "-", color="black", label="true")
        ax[0].plot(x_data_transformed[idx_], mean_network[idx_, 0], ".:", markevery=len(x_data_transformed)//21, color='red', label="approximated")
        ax[0].set_xlabel("Space")
        ax[0].set_ylabel("Drift")
        ax[0].legend()

        ax[1].plot(x_data_transformed[idx_], true_std_evaluated[idx_], "-", color="black", label="true")
        ax[1].plot(x_data_transformed[idx_], std_network[idx_, 0], ".:", markevery=len(x_data_transformed)//21, color='red', label="approximated")
        ax[1].set_xlabel("Space")
        ax[1].set_ylabel("Diffusivity")
        ax[1].legend()
    else:
        fig, ax = plt.subplots(2, n_dimensions, figsize=(n_dimensions * 3, 6))

        for k in range(n_dimensions):
            identity_pts = np.linspace(np.min([np.min(mean_network[:, k]), np.min(true_drift_evaluated)]),
                                       np.max([np.max(mean_network[:, k]), np.max(true_drift_evaluated)]),
                                       10)

            ax[0, k].scatter(mean_network[:, k], true_drift_evaluated[:, k], s=ms, label="approximation")
            ax[0, k].plot(identity_pts, identity_pts, 'k--', label="identity")
            ax[0, k].set_xlabel(r"network drift $f_" + str(k + 1) + "$")
            ax[0, k].set_ylabel(r"true drift $f_" + str(k + 1) + "$")
            ax[0, k].legend()

            identity_pts = np.linspace(np.min([np.min(std_network[:, k]), np.min(true_std_evaluated)]),
                                       np.max([np.max(std_network[:, k]), np.max(true_std_evaluated)]),
                                       10)

            ax[1, k].scatter(std_network[:, k], true_std_evaluated[:, k], s=ms, label="approximation")
            ax[1, k].plot(identity_pts, identity_pts, 'k--', label="identity")
            ax[1, k].set_xlabel(r"network diffusivity $\sigma_" + str(k + 1) + "$")
            ax[1, k].set_ylabel(r"true diffusivity $\sigma_" + str(k + 1) + "$")
            ax[1, k].legend()
    fig.tight_layout()
    return ax


def generate_results(apx_drift_diffusivity,
                     step_size, x_data, rng, T_steps=25, N_iterates=10,
                     data_transform=None,
                     data_backtransform=None):
    """
    x_data is used to sample initial conditions (N*p) matrix, with N initial conditions of dimension p.
    """
    if data_transform is None:
        data_transform = lambda x: x;
    if data_backtransform is None:
        data_backtransform = lambda x: x;

    def generate_path(f_sigma_, _x0, N):
        y_next = np.zeros((N, _x0.shape[1]))
        y_next[0, :] = _x0
        for k in range(1, N):
            y_next[k, :] = SDEIntegrators.euler_maruyama(y_next[k - 1, :].reshape(1, -1), step_size, f_sigma_, rng)
        return y_next

    time_steps = [np.arange(T_steps) * step_size] * N_iterates
    paths_ = []

    for k in range(N_iterates):
        path_ = generate_path(apx_drift_diffusivity, data_transform(x_data[k % x_data.shape[0], :].reshape(1, -1)),
                              T_steps)
        paths_.append(data_backtransform(path_))

    return time_steps, paths_


def plot_results_1d(time_steps, paths_network, paths_true, plot_dim=0, linewidth=1):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    if len(time_steps) == len(paths_network):
        time_steps = time_steps[0]

    for k in range(len(paths_network)):
        path_network = paths_network[k]
        path_true = paths_true[k]

        ax.plot(time_steps, path_network[:, plot_dim], 'r-', linewidth=linewidth,
                label='paths with network sde_model' if k == 0 else None);
        ax.plot(time_steps, path_true[:, plot_dim], 'k-', linewidth=linewidth,
                label='paths with SDE sde_model' if k == 0 else None);

    ax.set_xlabel("time")
    ax.set_ylabel("variable")

    ax.legend();
    fig.tight_layout()

    return fig


def probability_density(t, _generate_results, step_size, rng, plot_dim=0, N_iterates=10):
    """
    Samples the probability density of the given dimension at the given time.
    
    generate_results has to be a method from (step_size, T_steps, N_iterates) to (time_steps, paths_network, paths_true).
    Use generate_results_1d above as an example.
    """

    T_steps = int(t / step_size)

    time_steps, paths_ = _generate_results(step_size, T_steps=T_steps, N_iterates=N_iterates);

    pre_final_ = np.row_stack([pt[-2, :] for pt in paths_])
    final_ = np.row_stack([pt[-1, :] for pt in paths_])

    _pre_final_timestep = np.array([time_steps[k][-2] for k in range(len(time_steps))])
    _final_timestep = np.array([time_steps[k][-1] for k in range(len(time_steps))])

    final_ = np.row_stack([
        [
            np.poly1d(np.polyfit([_pre_final_timestep[k], _final_timestep[k]], [pre_final_[k, i], final_[k, i]], 1))(t)
            for i in range(pre_final_.shape[1])
        ]
        for k in range(len(_pre_final_timestep))
    ])

    for k in range(len(time_steps)):
        _t = time_steps[k]
        _t[-1] = t
        time_steps[k] = _t
    return final_, time_steps, paths_


def sinkhorn(points_A, points_B, n_iterations=5, kernel_scale=5e-2):
    """
    Computing the Sinkhorn algorithm between points a and b,
    as published by Cuturi (lightspeed optimal transport).
    """
    
    # squared pairwise distances with covariance = I
    _M = scipy.spatial.distance.cdist(points_A, points_B)**2

    local_kernel_scale = np.mean(_M)*kernel_scale

    _K = np.exp(-_M/local_kernel_scale)

    _x = np.ones_like(points_A)[:, :1]
    _r = np.ones_like(points_A)[:, :1]
    _c = np.ones_like(points_A)[:, :1]

    _x = _x / np.sum(_x)
    _r = _r / np.sum(_r)
    _c = _c / np.sum(_c)

    _U = _K * _M

    double_safe = 1e-15
    double_max = 1e5

    for i in range(n_iterations):
        _x = _r / np.clip(_K @ (_c / np.clip(np.transpose(_K) @ _x, double_safe, double_max)), double_safe, double_max)

    _v = _c / np.clip(np.transpose(_K) @ _x, double_safe, double_max)
    d_M = np.sum(_x * (_U @ _v))
    # _Plam = tf.linalg.diag(_u) @ _K @ tf.linalg.diag(_v)

    return d_M