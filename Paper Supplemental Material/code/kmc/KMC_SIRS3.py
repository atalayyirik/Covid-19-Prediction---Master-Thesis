

import numpy as np
import scipy
import matplotlib.pyplot as plt
from time import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

# to plot pretty histograms
import seaborn as sns
import pandas as pd

# our own code
from sde_sde_learning_network import \
    (
        ModelBuilder,
        GPModelSimple,
        SDEIdentification
    )

from kMClattice_KMC import *

GDLL_path1 = "kMCLattice.dll"

n_dimensions = 2

# Network parameters
n_latent_dimensions = 2
n_layers = 2
n_dim_per_layer = 50

LEARNING_RATE = 2e-4
BATCH_SIZE = 32
N_EPOCHS = 512
ACTIVATIONS = tf.nn.elu
VALIDATION_SPLIT = 0.1

# SIRS model paramters
k1,k2 = 1.0,1.0
k3 = 0.0
N1 = 32
N2 = 32
N = N1*N2

Dif1 = 50
Dif2 = 50

# training data parameters
n_trajectories = 4000
n_time_per_trajectory = 0.05
time_step = 0.01
Point1 = 1

random_state = 1

FileName = 'fig2a'
SaveAllData = False #True

n_steps = int(n_time_per_trajectory/time_step)
min_n_steps = 1
min_n_steps = n_steps
print("max_n_steps =", n_steps, ";  min_n_steps =", min_n_steps)

# parameters for NN and KMC paths at t = [0,tmax]
init_theta0 = 0.9
init_theta1 = 0.1
NPaths = 400

step_sizeNN = 2e-3
step_sizeKMC = 0.02
tmax = 3

# parameters for probability distribution at t = t_end
t_end = 2
N_iteratesNN = 10000
N_iteratesKMC = 10000

tf.random.set_seed(random_state)

# create KMC training data
print("initialize KMC")
mypath = os.path.dirname(os.path.realpath(__file__))
mypath = os.path.join(mypath, GDLL_path1)
print(mypath)
_SIR_KMC = SIR_KMC(random_state, N1, N2, k1, k2, k3, Dif1, Dif2, GDLL_path=mypath)
y0_all = []
t0 = time()
rng = np.random.default_rng(random_state)
for k in range(n_trajectories):
    NN1 = rng.integers(1, N)
    NN0 = rng.integers(0, N-NN1)
    y0_all.append([NN1/N, (N-NN0-NN1)/N])
time_g, y = _SIR_KMC.simulate(np.row_stack(y0_all), time_max = n_time_per_trajectory, step_size=time_step, Point1 = Point1)

x_data = []
y_data = []
step_sizes = []
for k in range(len(y)):
    y[k] = y[k][:,[1,2]]
    if len(time_g[k]) >= min_n_steps:
        step_sizes.extend(time_g[k][1:]-time_g[k][:-1])
        x_data.append(y[k][:-1,:])
        y_data.append(y[k][1:,:])

x_data = np.row_stack(x_data)
y_data = np.row_stack(y_data)
step_sizes = np.array(step_sizes)
print(f"sampling took {time()-t0} seconds.")
t0 = time()

def theta02(traj):    #Convert theta12 data to theta02 data (theta0 has "more interesting" trajectories)
    traj_new = traj.copy()
    traj_new[:,0] = 1-(traj[:,0]+traj[:,1])
    return traj_new

# work with (theta0,theta2) data instead of (theta1,theta2) data (the SDE equations we use to compare later are in terms of theta0,2)
x_data = theta02(x_data)
y_data = theta02(y_data)

# example encoder (Note: the encoder must output the mean and std for a Gaussian)
use_diag_std = True # because the true dynamics also is purely diagonal in the noise
model_type = "GP" # use a Gaussian process model to approximate the dynamics, instead of a VAE

if model_type=="GP":
    encoder = ModelBuilder.define_gaussian_process(
                                        n_input_dimensions=n_dimensions,
                                        n_output_dimensions=n_dimensions,
                                        n_layers=n_layers,
                                        n_dim_per_layer=n_dim_per_layer,
                                        name="GP",
                                        activation=ACTIVATIONS,
                                        use_diag_std=use_diag_std)
    model = GPModelSimple(encoder=encoder) # step size will be given during training


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))
#model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=1e-08, beta_1=0.9, beta_2=0.999))

sde_i = SDEIdentification(model=model)

print(x_data.shape)
print(y_data.shape)
y_full = np.column_stack([step_sizes, x_data, y_data])
NDataPoints = len(y_full)

hist = model.fit(x=y_full, epochs=N_EPOCHS, batch_size = BATCH_SIZE, verbose=2, validation_split=VALIDATION_SPLIT)

#model.summary()
TrainTime = time()-t0
print(f"NN was trained during  {TrainTime} seconds.")

N_iterates = NPaths
step_size = step_sizeNN 

# EXAMPLE SIR 1, for comparison of NN and KMC

# create sample paths for NN
initial_condition = np.array([init_theta0, 1.0-(init_theta0+init_theta1)]).reshape(1,-1)
T_steps = int(tmax/step_size)
paths_network = sde_i.sample_path(initial_condition, step_size, T_steps, N_iterates)
time_steps = [np.arange(T_steps+1) * step_size] * N_iterates
paths_network12 = [theta02(paths_network[k]) for k in range(len(paths_network))]

# create sample paths for KMC
y0 = np.ones((N_iterates,1)) @ initial_condition.reshape(1,-1)
times_test, y = _SIR_KMC.simulate(theta02(y0), time_max = tmax, step_size = step_sizeKMC, Point1 = 1)
y = [np.array(y[k][:,[1,2]]) for k in range(len(y))]
x_data_test = [theta02(y[k]) for k in range(len(y))]

# plot 3 figures for 3 paths of KMC and NN
def plot_results_1d(plot_dim=0, ytitle=""):
    fig,ax = plt.subplots(1,1, figsize=(6,4))
    for k in range(kPaths):
        if plot_dim <= 1:
            ax.plot(time_steps[k], paths_network[k][:,plot_dim], 'b-', linewidth=1.5, label='Network' if k==0 else None);
            ax.plot(times_test[k], x_data_test[k][:,plot_dim], 'r-', linewidth=1.0, label="KMC" if k==0 else None)
        else:
            ax.plot(time_steps[k], paths_network12[k][:,0], 'b-', linewidth=1.5, label='Network' if k==0 else None);
            ax.plot(times_test[k], theta02(x_data_test[k])[:,0], 'r-', linewidth=1.0, label="KMC" if k==0 else None)
    ax.set_xlabel("time")
    ax.set_ylabel(r"$\theta" + ytitle+"$")
    ax.set_ylim([0,1])
    ax.legend();
    fig.tight_layout()
    return fig

# plot 4 paths for Theta0,1,2 
kPaths = min(4, NPaths)
plot_results_1d(plot_dim=0, ytitle = "_0");
plot_results_1d(plot_dim=2, ytitle = "_1");
plot_results_1d(plot_dim=1, ytitle = "_2");

if SaveAllData == True: # save first 3 paths in a file
    f1 = open(FileName +'_NN_t.txt', 'w')
    f2 = open(FileName +'_KMC_t.txt', 'w')
    for i in range(len(paths_network[0][:, 0])):
        f1.write("%.8e  %.8e %.8e  %.8e %.8e  %.8e %.8e\n" % (time_steps[0][i], paths_network[0][i,0], paths_network[0][i,1], paths_network[1][i,0], paths_network[1][i,1], paths_network[2][i,0], paths_network[2][i,1]))
    for i in range(len(x_data_test[0][:, 0])):
        f2.write("%.8e  %.8e %.8e  %.8e %.8e  %.8e %.8e\n" % (times_test[0][i], x_data_test[0][i,0], x_data_test[0][i,1], x_data_test[1][i,0], x_data_test[1][i,1], x_data_test[2][i,0], x_data_test[2][i,1]))
    f1.close()
    f2.close()

# plot averaged paths for Theta0,1
def plot_results_mean(plot_dim=0, ytitle=""):
    if plot_dim < 1:
      ax[plot_dim].plot(time_steps[0], mpn[:,plot_dim], 'b-', linewidth=1.5, label='Mean of ' + str(NPaths) + ' NN paths');
      ax[plot_dim].plot(times_test[0], mpg[:,plot_dim], 'r-', linewidth=1.5, label='Mean of ' + str(NPaths) + ' KMC paths')
    else:
      ax[plot_dim].plot(time_steps[0], 1.0-mpn[:,0]-mpn[:,1], 'b-', linewidth=1.5, label='Mean of ' + str(NPaths) + ' NN paths');
      ax[plot_dim].plot(times_test[0], 1.0-mpg[:,0]-mpg[:,1], 'r-', linewidth=1.5, label='Mean of ' + str(NPaths) + ' KMC paths')
    ax[plot_dim].set_xlabel("time")
    ax[plot_dim].set_ylabel(r"$\theta" + ytitle+"$")
    ax[plot_dim].set_ylim([0,1])
    ax[plot_dim].legend();
    fig.tight_layout()


if np.array(x_data_test).ndim == 3:
    mpn = np.array(paths_network).mean(axis=0)
    mpg = np.array(x_data_test).mean(axis=0)
    fig,ax = plt.subplots(2,1, figsize=(10,6))
    plot_results_mean(plot_dim=0, ytitle = "_0");
    plot_results_mean(plot_dim=1, ytitle = "_1");
    plt.savefig(FileName +'_f4.png')
    if SaveAllData == True:
      f1 = open(FileName+'_NN_av1.txt', 'w')
      f2 = open(FileName+'_kmc_av1.txt', 'w')
      for i in range(len(paths_network[0][:, 0])):
        f1.write("%.8e %.8e %.8e\n" % (time_steps[0][i], mpn[i, 0], 1-mpn[i, 0]-mpn[i, 1]))
      for i in range(len(x_data_test[0][:, 0])):
        f2.write("%.8e %.8e %.8e\n" % (times_test[0][i], mpg[i, 0], 1-mpg[i, 0]-mpg[i, 1]))
      f1.close()
      f2.close()
else:
    print("some of the SSA paths are incomplete (try to decrease NPaths/tmax), the averaged paths cannot be constructed")

"""
Generate the probability density of all states at a particular time t.
"""

print("Sampling Network")
t0 = time()
N_iterates = N_iteratesNN
step_size = step_sizeNN
T_steps = int(t_end/step_size)
paths_n = sde_i.sample_path(initial_condition, step_size, T_steps, N_iterates)
time_steps = [np.arange(T_steps) * step_size] * N_iterates
pn = np.row_stack([pt[-1,:] for pt in paths_n])
print(f"sampling network took {time()-t0} seconds.")

print("Sampling KMC")
t0 = time()
N_iterates = N_iteratesKMC
y0 = np.ones((N_iterates,1)) @ initial_condition.reshape(1,-1)
time_steps_g, y = _SIR_KMC.simulate(theta02(y0), time_max = t_end, step_size=t_end, Point1 = 1)
y = [np.array(y[k][:,[1,2]]) for k in range(len(y))]
paths_g = [theta02(y[k]) for k in range(len(y))]
pg = np.row_stack([pt[-1,:] for pt in paths_g])
print(f"sampling KMC took {time()-t0} seconds.")

if SaveAllData == True:
    f1 = open(FileName+'_NN_data.txt', 'w')
    f2 = open(FileName+'_KMC_data.txt', 'w')
    for i in range(len(pn[:, 0])):
        f1.write("%.8e %.8e %.8e\n" % (pn[i, 0], 1-pn[i, 0]-pn[i, 1], pn[i, 1]))
    for i in range(len(pg[:, 0])):
        f2.write("%.8e %.8e %.8e\n" % (pg[i, 0], 1-pg[i, 0]-pg[i, 1], pg[i, 1]))
    f1.close()
    f2.close()

f2 = open(FileName+'.par', 'w')
f2.write("N = %d;  RS = %d\n" % (N, random_state))
f2.write("N_layers = %d;  N_dim = %d\n" % (n_layers, n_dim_per_layer))
f2.write("LEARNING_RATE = %.2e;  VALIDATION_SPLIT = %.2e" % (LEARNING_RATE, VALIDATION_SPLIT))
f2.write("  BATCH_SIZE = %d; N_EPOCHS = %d\n" % (BATCH_SIZE, N_EPOCHS))
f2.write("NTraj = %u;  t-max = %.2e;  t-step = %.2e;  NDataPoints = %d\n" % (n_trajectories, n_time_per_trajectory, time_step, NDataPoints))
f2.write("NIter = %u;  teta0 = %.2e;  teta1 = %.2e;  tmax = %.2e\n" % (N_iterates, init_theta0, init_theta1, t_end))
f2.write("min_n_steps = %d;  Point1 = %d\n" % (min_n_steps, Point1))
f2.write("k1 = %.2e;  k2 = %.2e;  k3 = %.2e\n" % (k1, k2, k3))
f2.write("Dif1 = %.2e;  Dif2 = %.2e\n" % (Dif1, Dif2))

f2.write("\nTrainTime = %.2e;  Adams\n" % (TrainTime))
f2.close()

#bins = np.linspace(0, 1, 50)
fig,ax = plt.subplots(3,1,figsize=(10,6))
kwargs = dict(hist_kws={'alpha':.3}, kde_kws={'linewidth':3})
def plot_Hist(NHist = 0, NTh = 0):
    if (NHist < 2):
        sns.distplot(pd.DataFrame(pn[:,NHist]),ax=ax[NHist], color="dodgerblue", label="Network", **kwargs);
        sns.distplot(pd.DataFrame(pg[:,NHist]),ax=ax[NHist], color="pink", label="KMC", **kwargs);
    else:
        sns.distplot(pd.DataFrame(1-pn[:,0]-pn[:,1]),ax=ax[NHist], color="dodgerblue", label="Network", **kwargs);
        sns.distplot(pd.DataFrame(1-pg[:,0]-pg[:,1]),ax=ax[NHist], color="pink", label="KMC", **kwargs);
    ax[NHist].set_xlabel(r"$\theta_"+str(NTh)+"(t="+str(t_end)+")$")
    ax[NHist].set_ylabel("density")
    ax[NHist].legend();

plot_Hist(0,0);
plot_Hist(1,2);
plot_Hist(2,1);

ax[0].set_xlim([0,1])
ax[1].set_xlim([0,1])
ax[2].set_xlim([0,1])

 #0.9,0.1 t=1.5
ax[0].set_xlim([0.0,0.4])
ax[1].set_xlim([0.25,0.65])
ax[2].set_xlim([0.15,0.55])

 #0.9,0.1 t=2
ax[0].set_xlim([0.0,0.2])
ax[1].set_xlim([0.5,0.7])
ax[2].set_xlim([0.2,0.4])

#ax[0].set_xlim([0.05,0.2])
#ax[1].set_xlim([0.5,0.65])
#ax[2].set_xlim([0.25,0.4])

fig.tight_layout()
plt.savefig(FileName +'.png')
plt.show()
plt.close() 



