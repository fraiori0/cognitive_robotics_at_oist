import jax
import jax.numpy as np
from jax import grad, jit, vmap


# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# derivative of the sigmoid function
def sigmoid_grad(x):
    return sigmoid(x) * (1 - sigmoid(x))


# tanh
def tanh(x):
    return np.tanh(x)


def tanh_grad(x):
    return 1 - np.tanh(x) ** 2


# sin function
def sin(x):
    return np.sin(x) + 1.0


def sin_grad(x):
    return np.cos(x)


# cos function
def cos(x):
    return np.cos(x) + 1.0


def cos_grad(x):
    return -np.sin(x)


# identity
def identity(x):
    return x


def identity_grad(x):
    return np.ones_like(x)


# loss function squared error
def loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).sum(axis=-1)


# derivative of the squared error loss function
def loss_grad(y_true, y_pred):
    return -2.0 * (y_true - y_pred)


class RNN:
    """Vanilla Recurrent Network with Sigmoid activation function."""

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        hidden_activation_fn=sigmoid,
        hidden_activation_fn_grad=sigmoid_grad,
        output_activation_fn=sigmoid,
        output_activation_fn_grad=sigmoid_grad,
        train_seq_length=10,
        seed=0,
        p_input_ratio=0.5,
    ):
        # network sizes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # training sequence length (useful to JIT)
        self.train_seq_length = train_seq_length
        # input ratio
        self.p_input_ratio = p_input_ratio

        # activation functions
        self.hidden_activation_fn = hidden_activation_fn
        self.hidden_activation_fn_grad = hidden_activation_fn_grad
        self.output_activation_fn = output_activation_fn
        self.output_activation_fn_grad = output_activation_fn_grad

        # random key
        self.key = jax.random.key(seed)

        # generate the weights by sampling from the normal distribution
        key_hh, key_xh, key_hy = jax.random.split(self.key, 3)

        self.params = {
            "hx": {
                "kernel_x": jax.random.normal(key_xh, (hidden_size, input_size)),
                "kernel_h": jax.random.normal(key_hh, (hidden_size, hidden_size)),
                "bias": np.zeros(hidden_size),
            },
            "y": {
                "kernel": jax.random.normal(key_hy, (output_size, hidden_size)),
                "bias": np.zeros(output_size),
            },
        }

        self.forward_train_single_jit = jit(self.forward_train_single)

    def forward_train_single(self, params, x, h0):
        """Forward pass for a single time step, used for JIT compilation.

        note we are missing the time dimension in the input x
        x.shape = (batch_size, input_size)
        h0.shape = (batch_size, hidden_size)

        Returns also intermediate values to implement backpropagation.
        """

        zs = {
            "x": x,
            "z_h": None,
            "z_x": None,
            "z_hx": None,
            "h": None,
            "z_y": None,
            "y": None,
        }

        z_x = (params["hx"]["kernel_x"] * x[:, None, :]).sum(axis=-1)
        z_h = (params["hx"]["kernel_h"] * h0[:, None, :]).sum(axis=-1)
        z_hx = z_x + z_h + params["hx"]["bias"]
        h = self.hidden_activation_fn(z_hx)

        z_y = (params["y"]["kernel"] * h[:, None, :]).sum(axis=-1)
        y = self.output_activation_fn(z_y + params["y"]["bias"])

        zs["z_h"] = z_h
        zs["z_x"] = z_x
        zs["z_hx"] = z_hx
        zs["h"] = h
        zs["z_y"] = z_y
        zs["y"] = y

        return zs

    def forward_train(self, key, params, x, h0):
        """Forward pass, apply the network recurrently for the train sequence length.

        x.shape = (batch_size, self.train_seq_length, input_size)
        h0.shape = (batch_size, hidden_size)

        Returns also intermediate values to implement backpropagation.
        """

        # store the intermediate values (before the activation function)
        zs = {
            # note we store even x because input ratio control
            # may give as input not the true value but the one self-generated
            "x": [],
            "z_h": [],
            "z_x": [],
            "z_hx": [],
            "h": [],
            "z_y": [],
            "y": [],
        }

        h = h0
        # append the initial hidden state
        # will be needed for the backpropagation
        zs["h"].append(h)

        # perform a first forward pass, after that we will apply input ratio control
        # (i.e., the first input x cannot be self-generated)
        zs_single = self.forward_train_single_jit(params, x[:, 0, :], h)
        for k, v in zs_single.items():
            zs[k].append(v)

        # loop forward in time
        for t in range(1, self.train_seq_length):

            # apply input ratio control
            # which is a probability of passing as input the true value
            # or the one self-generated by the network at the previous time-step
            # (in the latter case is like using already the feedback loop in training)
            key, _ = jax.random.split(key)
            mask = jax.random.bernoulli(key, self.p_input_ratio, (x.shape[0], 1))

            x_input = x[:, t, :] * mask + zs_single["y"] * (1 - mask)

            zs_single = self.forward_train_single_jit(params, x_input, h)

            for k, v in zs_single.items():
                zs[k].append(v)

        # convert the lists to arrays
        for k, v in zs.items():
            # stack along the time axis, shape will be (batch_size, train_seq_length, hidden_size)
            zs[k] = np.stack(v, axis=1)

        return zs

    def forward(self, params, x, h):
        """Forward pass, apply the network for one time step.

        x.shape = (batch_size, input_size)
        h.shape = (batch_size, hidden_size)

        """

        z_x = (params["hx"]["kernel_x"] * x[:, None, :]).sum(axis=-1)
        z_h = (params["hx"]["kernel_h"] * h[:, None, :]).sum(axis=-1)
        z_hx = z_x + z_h + params["hx"]["bias"]
        h = self.hidden_activation_fn(z_hx)

        z_y = (params["y"]["kernel"] * h[:, None, :]).sum(axis=-1)
        y = self.output_activation_fn(z_y + params["y"]["bias"])

        return y, h

    def backprop(self, params, zs, y_true):
        """Backward pass, compute the gradients of the loss with respect to the parameters.

        y_true.shape = (batch_size, self.train_seq_length, output_size)
        """

        # initialize the gradients, same shape as the parameters
        grads = {
            "hx": {
                "kernel_x": np.zeros_like(params["hx"]["kernel_x"]),
                "kernel_h": np.zeros_like(params["hx"]["kernel_h"]),
                "bias": np.zeros_like(params["hx"]["bias"]),
                # gradient on the first hidden state, passed to start the forward
                "h0": None,
            },
            "y": {
                "kernel": np.zeros_like(params["y"]["kernel"]),
                "bias": np.zeros_like(params["y"]["bias"]),
            },
        }

        # compute output errors at each time step
        # shape = (batch_size, train_seq_length, output_size)
        delta_y = loss_grad(y_true, zs["y"]) * self.output_activation_fn_grad(zs["z_y"])

        # initialize the hidden state error
        # shape = (batch_size, hidden_size), and then we compute one for each time step
        delta_h_t = np.zeros((y_true.shape[0], self.hidden_size))

        # iterate from the last time step to the first
        for t in reversed(range(self.train_seq_length)):
            # output error at the current time step
            delta_y_t = delta_y[:, t, :]
            # hidden state error at the current time step
            # note that zs["z_hx"][:, t, :] is the input to the sigmoid function, computed with
            # x(t) and h(t-1) on which we apply a sigmoid to compute h(t)
            delta_h_t = (
                (params["y"]["kernel"][None, :, :] * delta_y_t[:, :, None]).sum(axis=1)
                + (params["hx"]["kernel_h"][None, :, :] * delta_h_t[:, :, None]).sum(
                    axis=1
                )
            ) * self.hidden_activation_fn_grad(zs["z_hx"][:, t, :])

            # update the gradients
            # note, that zs["h"][:, t, :] is the hidden state at time step t-1
            # with respect to the current time step t, as we have the initial hidden state at the beginning
            grads["y"]["kernel"] = (
                grads["y"]["kernel"]
                + delta_y_t[:, :, None] * zs["h"][:, t + 1, None, :]  # note the t + 1
            )
            grads["y"]["bias"] = grads["y"]["bias"] + delta_y_t

            grads["hx"]["kernel_x"] = (
                grads["hx"]["kernel_x"] + delta_h_t[:, :, None] * zs["x"][:, t, None, :]
            )
            grads["hx"]["kernel_h"] = (
                grads["hx"]["kernel_h"]
                + delta_h_t[:, :, None] * zs["h"][:, t, None, :]  # note the t1
            )
            grads["hx"]["bias"] = grads["hx"]["bias"] + delta_h_t

        # gradient on the initial hidden state
        # that was given to start the forward pass
        # dims will be (batch_size, hidden_size)
        grads["hx"]["h0"] = (
            params["hx"]["kernel_h"][None, :, :] * delta_h_t[:, :, None]
        ).sum(axis=1)

        return grads

    def gen_hidden_state_normal(self, key, batch_size, scale=1.0):
        return jax.random.normal(key, (batch_size, self.hidden_size)) * scale

    def gen_hidden_state_uniform(self, key, batch_size, min, max):
        return jax.random.uniform(
            key, (batch_size, self.hidden_size), minval=min, maxval=max
        )

    def update_weights(self, params, grads, learning_rate):
        """Update the weights with the gradients using SGD.
        NOTE: this will not update the initial hidden state, as it's not a key of params
        """
        params = params.copy()
        for k in params.keys():
            for kk in params[k].keys():
                params[k][kk] = params[k][kk] - learning_rate * (grads[k][kk]).mean(
                    axis=0
                )

        return params
