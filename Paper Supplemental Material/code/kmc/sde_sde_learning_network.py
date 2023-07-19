
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras 
import tensorflow_probability as tfp
import numpy as np
import sys
# For numeric stability, set the default floating-point dtype to float64
tf.keras.backend.set_floatx('float64')
tfd = tfp.distributions


class ModelBuilder:
    """
    Constructs neural network models with specified topology.
    """
    @staticmethod
    def define_gaussian_process(n_input_dimensions, n_output_dimensions, n_layers, n_dim_per_layer, name, use_diag_std=True, activation="tanh", dtype=tf.float64):
        inputs = layers.Input((n_input_dimensions,), dtype=dtype, name=name + '_inputs')
        gp_x = inputs
        for i in range(n_layers):
            gp_x = layers.Dense(n_dim_per_layer, activation=activation, dtype=dtype, name=name + "_mean_hidden_{}".format(i))(gp_x)
        gp_output_mean = layers.Dense(n_output_dimensions, dtype=dtype, name=name + "_output_mean", activation=None)(gp_x)
        
        gp_x = inputs
        for i in range(n_layers):
            gp_x = layers.Dense(n_dim_per_layer, activation=activation, dtype=dtype, name=name + "_std_hidden_{}".format(i))(gp_x)

        if use_diag_std:
            gp_output_std = layers.Dense(n_output_dimensions, activation=lambda x: tf.nn.softplus(x) + 1e-7, name=name + "_output_std", dtype=dtype)(gp_x)
        else: # todo: the dimension of std should be N*(N+1)//2, not N^2, so that we can create the SPD matrix
            gp_output_std = layers.Dense((n_output_dimensions*n_output_dimensions), activation=None, name=name + "_output_std", dtype=dtype)(gp_x)
            
        gp = tf.keras.Model(inputs, [gp_output_mean, gp_output_std], name=name + "_gaussian_process")
        return gp


class GPModelSimple(tf.keras.Model):
    # This is a very simple model, essentially just a Gaussian process on x_n that predicts the drift and diffusivity

    def __init__(self, encoder: tf.keras.Model, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder

    def call(self, inputs):
        n_size = (inputs.shape[1]-1)//2
        step_size, x_n, x_np1 = tf.split(inputs, num_or_size_splits=[1, n_size, n_size], axis=1)
        approx_mean, approx_scale = self.encoder(x_n)

        # Explicitly coding the log probability of the normal distribution (Euler-Maruyama loss)
        _variance = step_size*tf.square(approx_scale)
        log_normal_probability = tf.square(x_np1 - (x_n + step_size * approx_mean)) /_variance
        self.add_loss(tf.math.reduce_mean(log_normal_probability + tf.math.log(_variance)))

        return approx_mean, approx_scale



class SDEIdentification:
    """
    Wrapper class that can be used for SDE identification.
    """
    def __init__(self, model):
        self.model = model

    def sample_path(self, x0, step_size, NT, N_iterates):
        """
        Use the neural network to sample a path with the Euler Maruyama scheme.
        """
        step_size = np.array(step_size).astype(np.float64)
        paths = [np.ones((N_iterates,1)) @ np.array(x0).reshape(1,-1)]
        for it in range(NT):
            x_n = paths[-1]
            apx_mean, apx_scale = self.model.encoder(x_n)

            x_np1 = tfd.MultivariateNormalDiag(
                loc=x_n + step_size * apx_mean,
                scale_diag=tf.math.sqrt(step_size) * apx_scale
            ).sample()
            
            paths.append(tf.keras.backend.eval(x_np1))
        return [ np.row_stack([paths[k][i] for k in range(len(paths))]) for i in range(N_iterates) ] 

        
