import tensorflow as tf
from tensorflow.keras import layers

import keras
import keras.backend as K

import tensorflow_probability as tfp

import sys
import numpy as np

tfd = tfp.distributions

# For numeric stability, set the default floating-point dtype to float64
tf.keras.backend.set_floatx('float64')
NUMBER_TYPE = tf.float64  # or tf.float32

STD_MIN_VALUE = 1e-13  # the minimal number that the diffusivity models can have


class SDEIntegrators:
    """
    Implements the common Euler-Maruyama and Milstein integrator
    schemes used in integration of SDE.
    """

    def __init__(self):
        pass

    @staticmethod
    def milstein(xn, h, _f_sigma, _sigma_gradient, rng):
        """
        Integration method for SDE, order 1 (strong) and 1 (weak) accurate.

        Parameters
        ----------
        xn current state
        h step size
        _f_sigma a lambda function that accepts a point x and returns a tuple (f(x), sigma(x))
        _sigma_gradient a lambda function that accepts a point x and returns the gradient of sigma
        rng a random number generator (e.g. numpy.random.default_rng(1))

        Returns
        -------
        x_{n+1}

        """
        dW1 = rng.normal(loc=0, scale=np.sqrt(h), size=xn.shape)
        xk = xn.reshape(1, -1)  # we only allow a single point as input

        drift_k, sigma_k = _f_sigma(xk)
        _sigma_gradient_evaluated = _sigma_gradient(xk)

        if not (np.prod(sigma_k.shape) == xk.shape[1]):
            raise ValueError("Diffusivity must be diagonal.")

        xnp1 = xk + h * drift_k + sigma_k * dW1 + 0.5 * sigma_k * _sigma_gradient_evaluated * (dW1 ** 2 - h)

        return xnp1

    @staticmethod
    def euler_maruyama(xn, h, _f_sigma, rng):
        """
        Integration method for SDE, order 1/2 (strong) and 1 (weak) accurate.

        Parameters
        ----------
        xn
        h
        _f_sigma
        rng

        Returns
        -------

        """
        dW = rng.normal(loc=0, scale=np.sqrt(h), size=xn.shape)
        xk = xn.reshape(1, -1)  # we only allow a single point as input

        fk, sk = _f_sigma(xk)
        if np.prod(sk.shape) == xk.shape[-1]:
            skW = sk * dW
        else:
            sk = sk.reshape(xk.shape[-1], xk.shape[-1])
            skW = (sk @ dW.T).T
        return xk + h * fk + skW



class ModelBuilder:
    """
    Constructs neural network models with specified topology.
    """
    DIFF_TYPES = ["diagonal", "triangular", "spd"]

    @staticmethod
    def define_forward_model(n_input_dimensions, n_output_dimensions, n_layers, n_dim_per_layer, name,
                             activation="tanh", dtype=tf.float64):

        inputs = layers.Input((n_input_dimensions,), dtype=dtype, name=name + '_inputs')
        network_x = inputs
        for i in range(n_layers):
            network_x = layers.Dense(n_dim_per_layer, activation=activation, dtype=dtype,
                                     name=name + "_hidden/dense_{}".format(i))(network_x)
        network_output = layers.Dense(n_output_dimensions, dtype=dtype,
                                      name=name + "_output_mean", activation=None)(network_x)

        network = tf.keras.Model(inputs=inputs, outputs=network_output,
                                 name=name + "_forward_model")
        return network

    @staticmethod
    def define_gaussian_process(n_input_dimensions, n_output_dimensions, n_layers, n_dim_per_layer, name,
                                diffusivity_type="diagonal", activation="tanh", dtype=tf.float64):
        
        def make_tri_matrix(z):
            # first, make all eigenvalues positive by changing the diagonal to positive values
            z = tfp.math.fill_triangular(z)
            z2 = tf.linalg.diag(tf.linalg.diag_part(z))
            z = z - z2 + tf.abs(z2)  # this ensures the values on the diagonal are positive
            return z

        def make_spd_matrix(z):
            z = make_tri_matrix(z)
            return tf.linalg.matmul(z, tf.linalg.matrix_transpose(z))
        
        inputs = layers.Input((n_input_dimensions,), dtype=dtype, name=name + '_inputs')
        gp_x = inputs
        for i in range(n_layers):
            gp_x = layers.Dense(n_dim_per_layer,
                                activation=activation,
                                dtype=dtype,
                                name=name + "_mean_hidden_{}".format(i))(gp_x)
        gp_output_mean = layers.Dense(n_output_dimensions, dtype=dtype,
                                      name=name + "_output_mean", activation=None)(gp_x)

        # initialize with extremely small (not zero!) values so that it does not dominate the drift
        # estimation at the beginning of training
        small_init = 1e-2
        initializer = tf.keras.initializers.RandomUniform(minval=-small_init, maxval=small_init, seed=None)

        gp_x = inputs
        for i in range(n_layers):
            gp_x = layers.Dense(n_dim_per_layer,
                                activation=activation,
                                dtype=dtype,
                                kernel_initializer=initializer,
                                bias_initializer=initializer,
                                name=name + "_std_hidden_{}".format(i))(gp_x)
        if diffusivity_type=="diagonal":
            gp_output_std = layers.Dense(n_output_dimensions,
                                         kernel_initializer=initializer,
                                         bias_initializer=initializer,
                                         activation=lambda x: tf.nn.softplus(x) + STD_MIN_VALUE,
                                         name=name + "_output_std", dtype=dtype)(gp_x)
        elif diffusivity_type=="triangular":
            # the dimension of std should be N*(N+1)//2, for one of the Cholesky factors L of the covariance,
            # so that we can create the lower triangular matrix with positive eigenvalues on the diagonal.
            gp_output_tril = layers.Dense((n_output_dimensions * (n_output_dimensions + 1) // 2),
                                          activation="linear",
                                          kernel_initializer=initializer,
                                          bias_initializer=initializer,
                                          name=name + "_output_cholesky", dtype=dtype)(gp_x)
            gp_output_std = layers.Lambda(make_tri_matrix)(gp_output_tril)
        elif diffusivity_type=="spd":
            # the dimension of std should be N*(N+1)//2, for one of the Cholesky factors L of the covariance,
            # so that we can create the SPD matrix C using C = L @ L.T to be used later.
            gp_output_tril = layers.Dense((n_output_dimensions * (n_output_dimensions + 1) // 2),
                                          activation="linear",
                                          kernel_initializer=initializer,
                                          bias_initializer=initializer,
                                          name=name + "_output_spd", dtype=dtype)(gp_x)
            gp_output_std = layers.Lambda(make_spd_matrix)(gp_output_tril)
            # gp_output_std = layers.Lambda(lambda L: tf.linalg.matmul(L, tf.transpose(L)))(gp_output_tril)
        else:
            raise ValueError(f"Diffusivity type {diffusivity_type} not supported. Use one of {ModelBuilder.DIFF_TYPES}.")
        
        gp = tf.keras.Model(inputs,
                            [gp_output_mean, gp_output_std],
                            name=name + "_gaussian_process")
        return gp


class LossAndErrorPrintingCallback(keras.callbacks.Callback):

    @staticmethod
    def __log(message, flush=True):
        sys.stdout.write(message)
        if flush:
            sys.stdout.flush()

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        LossAndErrorPrintingCallback.__log(
            "\rThe average loss for epoch {} is {:7.10f} ".format(
                epoch, logs["loss"]
            )
        )


class SDEIdentification:
    """
    Wrapper class that can be used for SDE identification.
    Needs a "tf.keras.Model" like the SDEApproximationNetwork or VAEModel to work.
    """

    def __init__(self, model):
        self.model = model

    def train_model(self, x_n, x_np1, validation_split=0.1, n_epochs=100, batch_size=1000, step_size=None,
                    callbacks=[]):
        print(f"training for {n_epochs} epochs with {int(x_n.shape[0] * (1 - validation_split))} data points"
              f", validating with {int(x_n.shape[0] * validation_split)}")

        if not (step_size is None):
            x_n = np.column_stack([step_size, x_n])
        y_full = np.column_stack([x_n, x_np1])

        if len(callbacks) == 0:
            callbacks.append(LossAndErrorPrintingCallback())

        hist = self.model.fit(x=y_full,
                              epochs=n_epochs,
                              batch_size=batch_size,
                              verbose=0,
                              validation_split=validation_split,
                              callbacks=callbacks)
        return hist

    def drift_diffusivity(self, x):
        drift, std = self.model.call_xn(x)
        return K.eval(drift), K.eval(std)

    def sample_path(self, x0, step_size, NT, N_iterates, map_every_iteration=None):
        """
        Use the neural network to sample a path with the Euler Maruyama scheme.
        """
        step_size = tf.cast(np.array(step_size), dtype=NUMBER_TYPE)
        paths = [np.ones((N_iterates, 1)) @ np.array(x0).reshape(1, -1)]
        for it in range(NT):
            x_n = paths[-1]
            apx_mean, apx_scale = self.model.call_xn(x_n)
            x_np1 = tfd.MultivariateNormalDiag(
                loc=x_n + step_size * apx_mean,
                scale_diag=tf.math.sqrt(step_size) * apx_scale
            ).sample()

            x_i = keras.backend.eval(x_np1)
            if not (map_every_iteration is None):
                x_i = map_every_iteration(x_i)
            paths.append(x_i)
        return [
            np.row_stack([paths[k][i] for k in range(len(paths))])
            for i in range(N_iterates)
        ]


class SDEApproximationNetwork(tf.keras.Model):
    """
    A neural network sde_model that uses a given
    sde_model network to predict drift and diffusivity
    of an SDE, and trains it using EM- or Milstein-scheme
    based loss functions.
    """
    VALID_METHODS = ["euler", "milstein", "milstein approx"]

    def __init__(self,
                 sde_model: tf.keras.Model,
                 step_size=None,
                 method="euler",
                 diffusivity_type="diagonal",
                 scale_per_point=1e-3,
                 **kwargs):
        super().__init__(**kwargs)
        self.sde_model = sde_model
        self.step_size = step_size
        self.method = method
        self.diffusivity_type = diffusivity_type
        self.scale_per_point = scale_per_point # only used if method="milstein approx"

        SDEApproximationNetwork.verify(self.method)

    @staticmethod
    def verify(method):
        if not (method in SDEApproximationNetwork.VALID_METHODS):
            raise ValueError(method + " is not a valid method. Use any of" + SDEApproximationNetwork.VALID_METHODS)

    def get_config(self):
        return {
            "sde_model": self.sde_model,
            "step_size": self.step_size,
            "method": self.method,
            "diffusivity_type": self.diffusivity_type
        }

    @staticmethod
    def milstein_pdf_regularized(ynp1_, yn_, h_, model_):
        """
        Computes the probability density of y(n+1) given y(n), of the Milstein method,
        for the sde

        dX_t = drift_(X_t) dt + diffusivity_(X_t) dB_t.

        Note: currently the wrong Bessel function is used (nu=0 instead of nu=-1/2),
        because TensorFlow does not have real-valued Bessel functions.

        Parameters
        ==========
        ynp1_:        next point
        yn_:          initial point
        h_:           step size
        model_:       drift and diffusivity in the sde

        Returns
        =======
        p(ynp1_ | yn_)  the probability density of y at the next point in time
        """
        with tf.GradientTape() as g:
            g.watch(yn_)
            drift_, diffusivity_ = model_(yn_)
        bprime = g.gradient(diffusivity_, yn_)
        #  bprime = g.batch_jacobian(diffusivity_, yn_)

        eps_safe = 1e-10
        zero_safe = 1e5
        bessel_safe = 1e5

        # this is CRUCIAL. If the derivative is not positive, then the numerical computations break down.
        bprime = tf.nn.softplus(bprime)

        A = yn_ + drift_ * h_ - 0.5 * diffusivity_ * bprime * h_
        B = diffusivity_ * tf.cast(tf.math.sqrt(h_), tf.float64)
        C = 0.5 * diffusivity_ * bprime * h_

        # make sure that the denominator is bounded away from zero, but in a smooth way!
        C = tf.math.softplus(C * zero_safe) / zero_safe + eps_safe

        nc_lambda = (B / (2 * C)) ** 2
        k_dof = 1
        x = (ynp1_ + (-A + (B ** 2) / (4 * C))) / C

        # x and nc_lambda also must never be zero or negative, since we take the square root later
        x_positive = tf.math.softplus(x * zero_safe) / zero_safe
        nc_lambda_positive = tf.math.softplus(nc_lambda * zero_safe) / zero_safe

        # also regularize the input to the bessel function
        bessel_input = tf.nn.tanh(tf.math.sqrt(x_positive * nc_lambda) / bessel_safe) * bessel_safe
        # bessel_input = tf.math.sqrt(x_positive * nc_lambda)

        # we use the bessel function of order 0 instead of -1/2 here... maybe a problem
        # this is the exponential version of the PDF, but we need the logarithmic version anyway
        # part_A = 0.5 * tf.math.exp(-(x+nc_lambda)/2)
        # part_B = tf.pow(x/nc_lambda, k_dof/4-1/2)
        # part_C = tf.math.bessel_i0(bessel_input) / C
        # p_xi = part_A + part_B + part_C

        part_A_log = tf.cast(tf.math.log(1 / 2), tf.float64) - (x + nc_lambda) / 2
        part_B_log = tf.cast(k_dof / 4 - 1 / 2, tf.float64) * tf.math.log(x_positive / nc_lambda_positive + eps_safe)
        part_C_log = tf.math.log(tf.math.bessel_i0(bessel_input)) - tf.math.log(C)
        p_xi_log = part_A_log + part_B_log + part_C_log

        return tf.reshape(p_xi_log, [-1])

    @staticmethod
    def milstein_pdf(ynp1_, yn_, h_, model_):
        """
        Computes the probability density of y(n+1) given y(n), of the Milstein method,
        for the sde

        dX_t = drift_(X_t) dt + diffusivity_(X_t) dB_t.

        Note: currently the wrong Bessel function is used (nu=0 instead of nu=-1/2),
        because TensorFlow does not have real-valued Bessel functions.

        Parameters
        ==========
        ynp1_:        next point
        yn_:          initial point
        h_:           step size
        model_:       drift and diffusivity in the sde

        Returns
        =======
        p(ynp1_ | yn_)  the probability density of y at the next point in time
        """
        with tf.GradientTape() as g:
            g.watch(yn_)
            drift_, diffusivity_ = model_(yn_)
        bprime = g.gradient(diffusivity_, yn_)
        #  bprime = g.batch_jacobian(diffusivity_, yn_)

        eps_safe = 1e-15
        zero_safe = 1e5
        bessel_safe = 1e5

        # this is CRUCIAL. If the derivative is not positive, then the numerical computations break down.
        bprime = tf.nn.softmax(bprime)

        A = yn_ + drift_ * h_ - 0.5 * diffusivity_ * bprime * h_
        B = diffusivity_ * tf.cast(tf.math.sqrt(h_), tf.float64)
        C = 0.5 * diffusivity_ * bprime * h_

        nc_lambda = (B / (2 * C)) ** 2
        k_dof = 1
        x = (ynp1_ + (-A + (B ** 2) / (4 * C))) / C

        # x and nc_lambda also must never be zero or negative, since we take the square root later
        x_positive = x
        nc_lambda_positive = nc_lambda

        # also regularize the input to the bessel function
        bessel_input = tf.math.sqrt(x_positive * nc_lambda_positive)

        # we use the bessel function of order 0 instead of -1/2 here... maybe a problem
        # this is the exponential version of the PDF, but we need the logarithmic version anyway
        # part_A = 0.5 * tf.math.exp(-(x+nc_lambda)/2)
        # part_B = tf.pow(x/nc_lambda, k_dof/4-1/2)
        # part_C = tf.math.bessel_i0(bessel_input) / C
        # p_xi = part_A + part_B + part_C

        part_A_log = tf.cast(tf.math.log(1 / 2), tf.float64) - (x_positive + nc_lambda_positive) / 2
        part_B_log = tf.cast(k_dof / 4 - 1 / 2, tf.float64) * tf.math.log(tf.math.abs(x_positive / nc_lambda_positive) + eps_safe)
        part_C_log = tf.math.log(tf.math.bessel_i0(bessel_input)) - tf.math.log(C)
        p_xi_log = part_A_log + part_B_log + part_C_log

        return tf.reshape(p_xi_log, [-1])

    @staticmethod
    def euler_maruyama_pdf(ynp1_, yn_, step_size_, model_, diffusivity_type="diagonal"):
        """
        This implies a very simple sde_model, essentially just a Gaussian process
        on x_n that predicts the drift and diffusivity.
        Returns log P(y(n+1) | y(n)) for the Euler-Maruyama scheme.

        Parameters
        ----------
        ynp1_ next point in time.
        yn_ current point in time.
        step_size_ step size in time.
        model_ sde_model that returns a (drift, diffusivity) tuple.
        diffusivity_type defines which type of diffusivity matrix will be used. See ModelBuilder.DIFF_TYPES.

        Returns
        -------
        logarithm of p(ynp1_ | yn_) under the Euler-Maruyama scheme.

        """
        drift_, diffusivity_ = model_(yn_)

        if diffusivity_type=="diagonal":
            approx_normal = tfd.MultivariateNormalDiag(
                loc=(yn_ + step_size_ * drift_),
                scale_diag=tf.math.sqrt(step_size_) * diffusivity_,
                name="approx_normal"
            )
        elif diffusivity_type=="triangular":
            diffusivity_tril_ = diffusivity_

            # a cumbersome way to multiply the step size scalar with the batch of matrices...
            # better use tfp.bijectors.FillScaleTriL()
            tril_step_size = tf.math.sqrt(step_size_)
            n_dim = K.shape(yn_)[-1]
            full_shape = n_dim * n_dim
            step_size_matrix = tf.broadcast_to(tril_step_size, [K.shape(step_size_)[0], full_shape])
            step_size_matrix = tf.reshape(step_size_matrix, (-1, n_dim, n_dim))

            # now form the normal distribution
            approx_normal = tfd.MultivariateNormalTriL(
                loc=(yn_ + step_size_ * drift_),
                scale_tril=tf.multiply(step_size_matrix, diffusivity_tril_),
                name="approx_normal"
            )
        elif diffusivity_type=="spd":
            diffusivity_spd_ = diffusivity_

            # a cumbersome way to multiply the step size scalar with the batch of matrices...
            # TODO: REFACTOR with diffusivity_type=="triangular"
            spd_step_size = tf.math.sqrt(step_size_) # NO square root because we use cholesky below?
            n_dim = K.shape(yn_)[-1]
            full_shape = n_dim * n_dim
            step_size_matrix = tf.broadcast_to(spd_step_size, [K.shape(step_size_)[0], full_shape])
            step_size_matrix = tf.reshape(step_size_matrix, (-1, n_dim, n_dim))

            # multiply with the step size
            covariance_matrix = tf.multiply(step_size_matrix, diffusivity_spd_)
            # square the matrix so that the cholesky decomposition does not change the eienvalues
            covariance_matrix = tf.linalg.matmul(covariance_matrix, tf.linalg.matrix_transpose(covariance_matrix))
            # perform cholesky to get the lower trianular matrix needed for MultivariateNormalTriL
            covariance_matrix = tf.linalg.cholesky(covariance_matrix)
            
            # now form the normal distribution
            approx_normal = tfd.MultivariateNormalTriL(
                loc=(yn_ + step_size_ * drift_),
                scale_tril=covariance_matrix,
                name="approx_normal"
            )
        else:
            raise ValueError(f"Diffusivity type <{diffusivity_type}> not supported. Use one of {ModelBuilder.DIFF_TYPES}.")
        return approx_normal.log_prob(ynp1_)

    @staticmethod
    def milstein_forward(eval_at_y_, yn_, h_, model_, scale_per_point=1e-3):
        """
        One forward pass of the Milstein method, implemented using TensorFlow.

        Parameters
        ----------
        yn_ current step
        h_ time step size at current step
        model_ keras.Model that returns a tuple of (drift,diffusivity matrix)

        Returns
        -------
        y(n+1) estimated using the Milstein scheme.
        """
        with tf.GradientTape() as g:
            g.watch(yn_)
            drift_, diffusivity_ = model_(yn_)
        diffusivity_prime_ = g.gradient(diffusivity_, yn_)

        approx_normal = tfd.MultivariateNormalDiag(
            loc=tf.zeros_like(yn_),
            scale_diag=tf.ones_like(yn_)
        )
        z = approx_normal.sample()
        sqrt_h = tf.cast(tf.math.sqrt(h_), dtype=tf.float64)
        ynp1 = yn_ + h_ * drift_ + diffusivity_ * sqrt_h * z + 0.5 * diffusivity_ * diffusivity_prime_ * (
                    h_ * z ** 2 - h_)

        kde_single_gaussian = tfd.Normal(loc=ynp1, scale=tf.cast(scale_per_point, dtype=tf.float64))
        return tf.reduce_sum(tf.map_fn(lambda y_: kde_single_gaussian.prob(y_), eval_at_y_), axis=1)

    @staticmethod
    def milstein_forward_approx(evaluate_at_y, yn_, h_, model_, scale_per_point=5e-3, num_estimations=100):
        """
        Estimation of the PDF of Milstein scheme, using a Gaussian Kernel Density estimate.

        Parameters
        ----------
        evaluate_at_y where to evaluate the pdf? Should be y(n+1)
        yn_ the given points y(n)
        h_ time step size at y(n)
        model_ keras.Model that returns (drift, diffusivity matrix) tuple
        scale_per_point kernel density estimator scale per point. Should scale with the number of points in yn_.
        num_estimations number of estimations, i.e. number of random numbers we draw from the Gaussian.

        Returns
        -------

        """
        log_eps = 1e-13  # to prevent numerical issues with the logarithm

        yn_repeated = tf.expand_dims(yn_, axis=0)
        yn_repeated = tf.tile(yn_repeated, (num_estimations, 1, 1))
        probabilities = tf.map_fn(lambda y_: SDEApproximationNetwork.milstein_forward(evaluate_at_y,
                                                                                      y_, h_, model_,
                                                                                      scale_per_point=scale_per_point),
                                  yn_repeated)

        return tf.math.log(tf.reduce_mean(probabilities, axis=0)+log_eps)

    @staticmethod
    def split_inputs(inputs, step_size=None):
        if step_size is None:
            n_size = (inputs.shape[1] - 1) // 2
            step_size, x_n, x_np1 = tf.split(inputs, num_or_size_splits=[1, n_size, n_size], axis=1)
        else:
            step_size = step_size
            x_n, x_np1 = tf.split(inputs, num_or_size_splits=2, axis=1)
        return step_size, x_n, x_np1

    def call_xn(self, inputs_xn):
        """
        Can be used to evaluate the drift and diffusivity
        of the sde_model. This is different than the "call" method
        because it only expects "x_k", not "x_{k+1}" as well.
        """
        return self.sde_model(inputs_xn)

    def call(self, inputs):
        """
        Expects the input tensor to contain all of (step_sizes, x_k, x_{k+1}).
        """
        step_size, x_n, x_np1 = SDEApproximationNetwork.split_inputs(inputs, self.step_size)

        if self.method == "euler":
            log_prob = SDEApproximationNetwork.euler_maruyama_pdf(x_np1, x_n, step_size, self.sde_model,
                                                                  self.diffusivity_type)
        elif self.method == "milstein":
            log_prob = SDEApproximationNetwork.milstein_pdf_regularized(x_np1, x_n, step_size, self.sde_model)
        elif self.method == "milstein approx":
            log_prob = SDEApproximationNetwork.milstein_forward_approx(x_np1, x_n, step_size, self.sde_model, self.scale_per_point)
        else:
            raise ValueError(self.method + " not available")

        sample_distortion = -tf.reduce_mean(log_prob, axis=-1)
        distortion = tf.reduce_mean(sample_distortion)

        loss = distortion

        self.add_loss(loss)
        self.add_metric(distortion, name="distortion", aggregation="mean")

        return self.sde_model(x_n)


class VAEModel(tf.keras.Model):
    """
    This is an auto-encoder sde_model that defines a Gaussian process over an encoding space created from x_n,
    essentially predicting x_np1 through drift and diffusivity over the encoding, not the input.
    This allows to reduce the dimension of the space, if we are given (too many) observations of an SDE process.
    """

    def __init__(self,
                 encoder: tf.keras.Model,
                 decoder: tf.keras.Model,
                 latent_sde_model: tf.keras.Model,
                 step_size=None,
                 diffusivity_type="diagonal",
                 method="euler",
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.encoder_diffusivity = tf.Variable(initial_value=1e-2, trainable=False, dtype=NUMBER_TYPE)
        self.decoder = decoder
        self.latent_sde_model = latent_sde_model
        self.step_size = step_size
        self.diffusivity_type = diffusivity_type
        self.method = method

        SDEApproximationNetwork.verify(self.method)

    def get_config(self):
        return dict(
            encoder=self.encoder,
            encoder_diffusivity=self.encoder_diffusivity,
            decoder=self.decoder,
            step_size=self.step_size,
            diffusivity_type=self.diffusivity_type
        )

    def call_xn(self, inputs_xn):
        """
        Similar to SDEApproximationNetwork.call_xn, calls decoder(encoder(inputs)).

        Parameters
        ----------
        inputs x_n points

        Returns
        -------
        (drift, diffusivity) tuple, evaluated at x_n
        """
        return self.latent_sde_model(self.encoder(inputs_xn))

    def call(self, inputs):
        """
        Expects the input tensor to contain all of (step_sizes, x_k, x_{k+1}).
        """
        step_size, x_n, x_np1 = SDEApproximationNetwork.split_inputs(inputs, self.step_size)

        # Model distributions in the latent space
        apx_post_mean = self.encoder(x_n)
        apx_post = tfd.Independent(tfd.MultivariateNormalDiag(
            loc=apx_post_mean,
            scale_diag=None,
            scale_identity_multiplier=self.encoder_diffusivity,
            name="apx_post"
        ), reinterpreted_batch_ndims=0)
        prior = tfd.Independent(tfd.MultivariateNormalDiag(
            loc=tf.zeros_like(apx_post_mean),
            scale_diag=tf.ones_like(apx_post_mean),
            scale_identity_multiplier=None,
            name="prior"
        ), reinterpreted_batch_ndims=0)

        def full_model(_input):
            """
            Needed in addition to call_xn because it has to be converted to a tf.function.
            Parameters
            ----------
            _input

            Returns
            -------

            """
            return self.latent_sde_model(self.encoder(_input))

        if self.method == "euler":
            log_prob = SDEApproximationNetwork.euler_maruyama_pdf(x_np1, x_n, step_size, full_model,
                                                                  self.diffusivity_type)
        else:
            raise ValueError(self.method + " not available in VAEModel")

        # Distortion
        sample_distortion = tf.reduce_mean(tf.square(x_n-self.decoder(apx_post.sample())), axis=-1)
        distortion = tf.math.log(tf.reduce_mean(sample_distortion))

        # Rate
        sample_rate = tfp.distributions.kl_divergence(apx_post, prior)
        rate = tf.math.log(tf.reduce_mean(sample_rate))
        
        # SDE Distortion
        sde_sample_distortion = -tf.reduce_mean(log_prob, axis=-1)
        sde_distortion = tf.reduce_mean(sde_sample_distortion)

        # Loss
        loss = distortion + sde_distortion + rate

        # Add loss and metrics
        self.add_loss(loss)
        self.add_metric(rate, name="rate", aggregation="mean")
        self.add_metric(distortion, name="distortion", aggregation="mean")
        self.add_metric(sde_distortion, name="sde_distortion", aggregation="mean")

        return x_np1

    @staticmethod
    def define_model(n_input_dimensions, n_latent_dimensions, n_layers, n_dim_per_layer,
                     diffusivity_type="diagonal", activation="tanh"):
        encoder = ModelBuilder.define_forward_model(n_input_dimensions,
                                                    n_latent_dimensions,
                                                    n_layers,
                                                    n_dim_per_layer,
                                                    "encoder",
                                                    activation=activation)
        decoder = ModelBuilder.define_gaussian_process(n_latent_dimensions,
                                                       n_input_dimensions,
                                                       n_layers,
                                                       n_dim_per_layer,
                                                       "decoder",
                                                       activation=activation,
                                                       diffusivity_type=diffusivity_type)

        def full_model(x):
            return decoder(encoder(x))

        return encoder, decoder, full_model


class AEModel(tf.keras.Model):
    """
    This is an auto-encoder sde_model that defines a Gaussian process over an encoding space created from x_n,
    essentially predicting x_np1 through drift and diffusivity over the encoding, not the input.
    This allows to reduce the dimension of the space, if we are given (too many) observations of an SDE process.
    """

    def __init__(self,
                 encoder: tf.keras.Model,
                 decoder: tf.keras.Model,
                 latent_sde_model: tf.keras.Model,
                 step_size=None,
                 diffusivity_type="diagonal",
                 method="euler",
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.latent_sde_model = latent_sde_model
        self.step_size = step_size
        self.diffusivity_type = diffusivity_type
        self.method = method

        SDEApproximationNetwork.verify(self.method)

    def get_config(self):
        return dict(
            encoder=self.encoder,
            decoder=self.decoder,
            step_size=self.step_size,
            diffusivity_type=self.diffusivity_type
        )

    def call_xn(self, inputs_xn):
        """
        Similar to SDEApproximationNetwork.call_xn, calls decoder(encoder(inputs)).

        Parameters
        ----------
        inputs x_n points

        Returns
        -------
        (drift, diffusivity) tuple, evaluated at x_n
        """
        return self.latent_sde_model(self.encoder(inputs_xn))

    def call(self, inputs):
        """
        Expects the input tensor to contain all of (step_sizes, x_k, x_{k+1}).
        """
        step_size, x_n, x_np1 = SDEApproximationNetwork.split_inputs(inputs, self.step_size)

        def full_model(_input):
            """
            Needed in addition to call_xn because it has to be converted to a tf.function.
            Parameters
            ----------
            _input

            Returns
            -------

            """
            return self.latent_sde_model(self.encoder(_input))

        if self.method == "euler":
            log_prob = SDEApproximationNetwork.euler_maruyama_pdf(x_np1, x_n, step_size, full_model,
                                                                  self.diffusivity_type)
        else:
            raise ValueError(self.method + " not available in VAEModel")

        # Distortion
        sample_distortion = tf.reduce_mean(tf.square(x_n - self.decoder(self.encoder(x_n))), axis=-1)
        distortion = tf.math.log(tf.reduce_mean(sample_distortion))
        
        # SDE Distortion
        sde_sample_distortion = -tf.reduce_mean(log_prob, axis=-1)
        sde_distortion = tf.reduce_mean(sde_sample_distortion)

        # Add loss and metrics
        self.add_loss(distortion + sde_distortion)
        self.add_metric(distortion, name="distortion", aggregation="mean")
        self.add_metric(sde_distortion, name="sde_distortion", aggregation="mean")

        return x_np1
