from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
import keras.backend as K

import scipy.spatial

## pip install "git+https://github.com/google/edward2.git#egg=edward2"
# import edward2 as ed
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
import numpy as np


tf.keras.backend.set_floatx('float64')
    
class WassersteinSinkhornModel(tf.keras.Model):
    """
    A sde_model that represents the Wasserstein transport between two probability distributions.
    """

    def __init__(self,
                 transport: tf.keras.Model,
                 transport_inverse: tf.keras.Model = None,
                 kernel_scale: np.float = 5e-2,
                 change_tolerance : np.float = 1e-8,
                 n_iterations: np.int = 20,
                 **kwargs):
        super().__init__(**kwargs)  # dynamic=True, 
        self.n_iterations = n_iterations
        self.change_tolerance = change_tolerance
        self.transport = transport
        self.transport_inverse = transport_inverse
        self.kernel_scale_user = kernel_scale
        
    @staticmethod
    def sinkhorn(points_A, points_B, n_iterations=3, kernel_scale=5e-2):
        """
        Implement the fast Sinkhorn algorithm from Cuturi, from here: 
        http://dl.acm.org/citation.cfm?id=2999792.2999868
        
        Note that "length_scale" is lambda, for lambda in the paper, and the cost is
        the squared Euclidean distance.
        Here, we use point clouds as input and set r and c to be constant.
        
        Input M, lambda, r, c.
            I=(r>0); r=r(I); M=M(I,:); K=exp(-lambda*M)
            Set x=ones(length(r),size(c,2))/length(r);
            while x changes do
                x=diag(1./r)*K*(c.*(1./(K'*(1./x))))
            end while
            u=1./x; v=c.*(1./(K'*u))
            d_M(r,c)=sum(u.*((K.*M)*v))
        """
        
        # squared pairwise distances with covariance = I
        pairwise_diff = K.expand_dims(points_A, 0) - K.expand_dims(points_B, 1)
        pairwise_squared_distance = K.sum(K.square(pairwise_diff), axis=-1)
        _M = pairwise_squared_distance
        
        local_kernel_scale = K.mean(_M)*kernel_scale
            
        _K = tf.math.exp(-_M/local_kernel_scale)
        
        _x = tf.ones_like(points_A)[:,:1]
        _r = tf.ones_like(points_A)[:,:1]
        _c = tf.ones_like(points_A)[:,:1]
        
        _x = _x / tf.reduce_sum(_x)
        _r = _r / tf.reduce_sum(_r)
        _c = _c / tf.reduce_sum(_c)
        
        _U = _K * _M
        
        _xold = _x
        
        double_safe = 1e-15
        double_max = 1e5
        
        for i in range(n_iterations):
            _x = _r / tf.clip_by_value(_K @ (_c / tf.clip_by_value(tf.transpose(_K) @ _x, double_safe, double_max)), double_safe, double_max)
            
        _v = _c / tf.clip_by_value(tf.transpose(_K) @ _x, double_safe, double_max)
        d_M = tf.reduce_sum(_x * (_U @ _v))
        # _Plam = tf.linalg.diag(_u) @ _K @ tf.linalg.diag(_v)
        
        return d_M

    def call(self, inputs):
        # split the batch into source and target domain sets
        batch_source, batch_target = tf.split(inputs, num_or_size_splits=2, axis=1)

        approx_target = self.transport(batch_source)

        # call sinkhorn
        wasserstein_distance = self.sinkhorn(batch_target, approx_target,
                                             n_iterations=self.n_iterations,
                                             kernel_scale=self.kernel_scale_user)
        
        loss = wasserstein_distance
        
        self.add_metric(wasserstein_distance, name="wasserstein_distance", aggregation="mean")
        
        if not(self.transport_inverse is None):
            approx_source = self.transport_inverse(approx_target)
            identity_distance = tf.reduce_mean(tf.square(approx_source-batch_source))
            loss += identity_distance

            self.add_metric(identity_distance, name="inverse_distance", aggregation="mean")

        self.add_loss(loss)
        
        return approx_target
    
    
class WassersteinLipschitzModel(tf.keras.Model):
    """
    A sde_model that represents the Wasserstein transport between two probability distributions.
    """

    def __init__(self,
                 transport: tf.keras.Model,
                 transport_lipschitz: tf.keras.Model,
                 transport_inverse: tf.keras.Model,
                 **kwargs):
        super().__init__(**kwargs)
        self.transport = transport
        self.transport_inverse = transport_inverse
        self.lipschitz = transport_lipschitz
        self.train_transporter = True

    def call(self, inputs):
        # split the batch into source and target domain sets
        batch_source, batch_target = tf.split(inputs, num_or_size_splits=2, axis=1)

        approx_target = self.transport(batch_source)
        #approx_source = self.transport_inverse(approx_target)

        # call lipschitz function to measure difference
        score_approx = self.lipschitz(approx_target)
        score_true = self.lipschitz(batch_target)
        #identity_loss = tf.reduce_mean(tf.square(approx_source - batch_source))
        
        if self.train_transporter:
            loss = -tf.reduce_mean(score_approx)# + identity_loss
        else:
            loss = -(tf.reduce_mean(score_true) - tf.reduce_mean(score_approx))
        
        self.add_metric(score_approx, name="score_approx", aggregation="mean")
        self.add_metric(score_approx, name="score_true", aggregation="mean")
        #self.add_metric(identity_loss, name="identity_loss", aggregation="mean")
        self.add_loss(loss)
        
        return approx_target