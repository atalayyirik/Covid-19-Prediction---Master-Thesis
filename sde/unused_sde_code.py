import tensorflow as tf
from tensorflow.keras import layers

import keras
import keras.backend as K

import tensorflow_probability as tfp

import sys
import numpy as np

from .wasserstein_networks import WassersteinSinkhornModel


# For numeric stability, set the default floating-point dtype to float64
tf.keras.backend.set_floatx('float64')


    
    

def euler_maruyama_sampler(y_n, y_np1, f_network, sigma_network, h):
    # TODO: at the moment, sigma_network can only be a diagonal array (i.e. no full SPD matrix.)
    """
    Reformulation of the Euler Maruyama scheme in terms of the noise process:
    EM: x_{n+1} = x_n + h f(x_n) + sqrt(h) * sigma(x_n) * dW, where dW \sim N(0,1).
    
    If f and sigma are NOT the correct drift and diffusivity (which they are not, since we are approximating them by f_net and sigma_net), and we do not know the noise process dW but an identically distributed one dQ, then:
    x_{n+1} - x_n = h f(x_n) + sqrt(h) * sigma(x_n) * dW
    and thus
    
    (x_{n+1} - x_n - h f_net(x_n)) / (sqrt(h) * sigma_net(x_n)) = 
    (h f(x_n) + sqrt(h) * sigma(x_n) * dW - h f_net(x_n)) / (sqrt(h) * sigma_net(x_n))
    
    which has mean 
        sqrt(h) (f(x_n) - f_net(x_n)) / sigma_net(x_n)
    and variance
        VAR(sigma(x_n)/sigma_net(x_n)  * dW) = sigma(x_n)^2/sigma_net(x_n)^2
        
    
    """
    return (y_np1 - (y_n + h * f_network)) / (tf.math.sqrt(h) * sigma_network)

        
class EMSamplerModel(tf.keras.Model):
    """
    This sde_model uses the EM sampler to construct the log probability of the multivariate
    standard normal as the SDEApproximationNetwork does using the probability framework.
    """

    def __init__(self,
                 encoder: tf.keras.Model,
                 expectation_scale = 1e3,
                 step_size = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.expectation_scale = expectation_scale
        self.step_size = step_size

    def call_xn(self, inputs_xn):
        return self.encoder(inputs_xn)

    def call(self, inputs):
        if self.step_size is None:
            n_size = (inputs.shape[1]-1)//2
            step_size, x_n, x_np1 = tf.split(inputs, num_or_size_splits=[1, n_size, n_size], axis=1)
        else:
            step_size = self.step_size
            x_n, x_np1 = tf.split(inputs, num_or_size_splits=2, axis=1)

        approx_mean, approx_scale = self.encoder(x_n)

        batch_samples = euler_maruyama_sampler(x_n, x_np1, approx_mean, approx_scale, step_size)
        
        # compute the point-wise loss
        _variance = tf.square(tf.math.sqrt(step_size) * approx_scale)
        log_normal_probability = -0.5 * tf.square(batch_samples)
        # with the regularization based on the standard deviation
        log_regularizer = -0.5 * tf.math.log(_variance)
        
        distortion_sample = -tf.reduce_sum(log_normal_probability+log_regularizer, axis=-1)
        distortion = tf.reduce_mean(distortion_sample)
        
        loss = distortion

        self.add_loss(loss)
        self.add_metric(distortion, name="distortion", aggregation="mean")

        return approx_mean, approx_scale
    
class FrequentistModel(tf.keras.Model):
    """
    Tests on normality of the output, after transforming with the euler_maruyama_sampler scheme.
    """
    def __init__(self,
                 model: tf.keras.Model,
                 step_size = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model
        if not(step_size is None):
            step_size = np.array(step_size)
        self.step_size = step_size

    def call_xn(self, inputs_xn):
        return self.model(inputs_xn)

    def call(self, inputs):
        if self.step_size is None:
            n_size = (inputs.shape[1]-1)//2
            step_size, x_n, x_np1 = tf.split(inputs, num_or_size_splits=[1, n_size, n_size], axis=1)
        else:
            step_size = self.step_size
            x_n, x_np1 = tf.split(inputs, num_or_size_splits=2, axis=1)

        approx_mean, approx_scale = self.model(x_n)

        # compute the EM scheme that is hopefully N(0,1) distributed
        batch_samples = euler_maruyama_sampler(x_n, x_np1, approx_mean, approx_scale, step_size)
        frequentist = FrequentistModel.test_normality_tensor(batch_samples)

        # test normality of the sample
        loss = frequentist
        self.add_loss(loss)
        self.add_metric(frequentist, name="frequentist", aggregation="mean")

        return approx_mean, approx_scale

    @staticmethod
    def test_normality_tensor(batch_samples):
        """
        test_normality of the given samples.
        We only test w.r.t. the standard normal, not for "any" normal distribution.

        Based on: https://doi.org/10.1007/BF02613322

        Parameters
        ----------
        batch_samples
            N x d matrix of N points in d dimensions.
        Returns
        -------
        Test statistic Tn from the paper.
        """

        # squared pairwise distances with covariance = I
        pairwise_diff = K.expand_dims(batch_samples, 0) - K.expand_dims(batch_samples, 1)
        pairwise_squared_distance = K.sum(K.square(pairwise_diff), axis=-1)
        Rjk = pairwise_squared_distance

        # squared distances to mean zero
        Rj2 = K.sum(K.square(batch_samples), axis=-1)
        
        dtype = batch_samples.dtype

        # number of samples
        batch_size = K.cast(K.shape(batch_samples)[0], dtype=dtype)
        # dimension of samples
        n_dimensions = K.cast(K.shape(batch_samples)[1], dtype=dtype)

        # test statistic
        # Tn0 = K.mean(K.exp(-0.5*K.flatten(Rjk))) # for some reason this gives different results than the next line
        Tn0 = tf.cast(1.0, dtype) / batch_size * K.sum(K.exp(-0.5 * K.flatten(Rjk)))
        Tn1 = -K.pow( tf.cast(2.0, dtype) , 1 - n_dimensions / tf.cast(2.0, dtype)) * K.sum(K.exp(-0.25 * Rj2))
        Tn2 = batch_size * K.pow(tf.cast(3.0, dtype), -n_dimensions / tf.cast(2.0, dtype))
        Tn = Tn0 + Tn1 + Tn2

        return Tn
    

class WassersteinSDEModel(tf.keras.Model):
    """
    Tests on wasserstein distance of the output to the standard normal,
    after transforming with the euler_maruyama_sampler scheme.
    """
    def __init__(self,
                 model: tf.keras.Model,
                 step_size = None,
                 random_state = 1,
                 kernel_scale = 5e-2,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.step_size = step_size
        self.kernel_scale = kernel_scale
        self.random_state = random_state
        self._rng = tf.random.Generator.from_seed(random_state)

    def call_xn(self, inputs_xn):
        return self.model(inputs_xn)

    def call(self, inputs):
        if self.step_size is None:
            n_size = (inputs.shape[1]-1)//2
            step_size, x_n, x_np1 = tf.split(inputs, num_or_size_splits=[1, n_size, n_size], axis=1)
        else:
            step_size = self.step_size
            x_n, x_np1 = tf.split(inputs, num_or_size_splits=2, axis=1)

        approx_mean, approx_scale = self.model(x_n)

        # compute the EM scheme that is hopefully N(0,1) distributed
        batch_samples = euler_maruyama_sampler(x_n, x_np1, approx_mean, approx_scale, step_size)
        normal_samples = self._rng.normal(mean=0, stddev=1, shape=K.shape(batch_samples), dtype=batch_samples.dtype)

        # compute the wasserstein distance to the standard normal
        wasserstein_distance_to_normal = WassersteinSinkhornModel.sinkhorn(batch_samples, normal_samples, kernel_scale = self.kernel_scale)
        
        # combine sample and point-wise losses
        loss = wasserstein_distance_to_normal
        self.add_loss(loss)
        self.add_metric(wasserstein_distance_to_normal, name="wasserstein_distance_to_normal", aggregation="mean")

        return approx_mean, approx_scale