import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

from sde.sde_learning_network import \
    (
        SDEIdentification,
        ModelBuilder,
        SDEApproximationNetwork,
    )

from sde.experiment_reports import \
(
    sample_data,
    plot_results_functions,
    generate_results,
    plot_results_1d
)
from sde.thesis_experiments import \
(
    loadSIRInfo,
    loadSIRInfo1D
)

# notebook parameters
random_seed = 1
step_size = 1e-1  # 5e-2 # step size
n_pts = 5000        # number of points

n_layers = 5
n_dim_per_layer = 25

n_dimensions = 1

ACTIVATIONS = tf.nn.sigmoid
VALIDATION_SPLIT = .2
BATCH_SIZE = 512
N_EPOCHS = 20

# only diagonal diffusivity matrix (does not matter since we are in 1D)
diffusivity_type = "diagonal"

tf.random.set_seed(random_seed)

x_data,y_data,x_data_test,y_data_test = loadSIRInfo1D()

step_sizes = np.zeros((x_data.shape[0],)) + step_size

# define the neural network model we will use for identification
encoder = ModelBuilder.define_gaussian_process(
                                        n_input_dimensions=n_dimensions,
                                        n_output_dimensions=n_dimensions,
                                        n_layers=n_layers,
                                        n_dim_per_layer=n_dim_per_layer,
                                        name="GP",
                                        activation=ACTIVATIONS,
                                        diffusivity_type=diffusivity_type)
encoder.summary()

model = SDEApproximationNetwork(sde_model=encoder, method="euler")
model.compile(optimizer=tf.keras.optimizers.Adamax())

sde_i = SDEIdentification(model=model)

hist = sde_i.train_model(x_data, y_data, step_size=step_sizes,
                         validation_split=VALIDATION_SPLIT, n_epochs=N_EPOCHS, batch_size = BATCH_SIZE)

T_steps = 30
N_iterates = 1

rng = np.random.default_rng(random_seed)

time_steps, paths_network = \
    generate_results(sde_i.drift_diffusivity,
                     step_size, x_data_test, rng,
                     T_steps=T_steps, N_iterates=N_iterates);

x_data_test_new = np.zeros(shape=x_data_test.shape) 
for i in range(len(x_data_test)):
    if i%4==0:
        x_data_test_new[i//4] = x_data_test[i]

true_path = []
for i in range(0, len(x_data_test), 4):
    path = np.array([x_data_test[i], x_data_test[i+1], x_data_test[i+2],x_data_test[i+3],y_data_test[i+3]])
    true_path.append(path)

from sde.sde_learning_network import \
(
    SDEIntegrators
)
def generate_results2(apx_drift_diffusivity,
                     step_size, x_data, rng, T_steps=25, N_iterates=10,                     
                     p_data=None):
    """
    x_data is used to sample initial conditions (N*p) matrix, with N initial conditions of dimension p.
    """

    def generate_path(f_sigma_, _x0, N, _p0=None):
        y_next = np.zeros((N, _x0.shape[1]))
        y_next[0, :] = _x0
        for k in range(1, N):
            _y_k = y_next[k - 1, :].reshape(1, -1)
            _p_k = _p0
            y_next[k, :] = SDEIntegrators.euler_maruyama(_y_k,
                                                         step_size,
                                                         f_sigma_,
                                                         rng,
                                                         param=_p_k)
        return y_next

    time_steps = [np.arange(T_steps) * step_size] * N_iterates
    paths_ = []

    p0 = None
    for k in range(N_iterates):
        #print(x_data[k])
        x0 = x_data[k].reshape(1, -1)
        path_ = generate_path(apx_drift_diffusivity, x0,
                              T_steps, _p0=p0)
        #print(path_)
        paths_.append(path_)

    return time_steps, paths_

        
        

time_steps, paths_network = generate_results2(sde_i.drift_diffusivity,
                     step_size, x_data_test_new, rng,
                     T_steps=2, N_iterates=1);
print(paths_network)