import jax
import jax.numpy as np
from jax import grad, jit, vmap


# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# derivative of the sigmoid function
def sigmoid_grad(x):
    return sigmoid(x) * (1 - sigmoid(x))


# loss function squared error
def se(y_true, y_pred):
    return ((y_true - y_pred) ** 2).sum(axis=-1)


# derivative of the squared error loss function
def se_grad(y_true, y_pred):
    return -2.0 * (y_true - y_pred)


class RNN:
    """ Vanilla Recurrent Network with Sigmoid activation function. """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        train_seq_length=10,
        seed=0,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.train_seq_length = train_seq_length

        self.key = jax.random.key(seed)

        # generate the weights by sampling from the normal distribution
        key_hh, key_xh, key_hy = jax.random.split(self.key, 3)

        self.params = {
            "hhx": {
                "kernel_x": jax.random.normal(key_xh, (hidden_size, input_size)),
                "kernel_h": jax.random.normal(key_hh, (hidden_size, hidden_size)),
                "bias": np.zeros(hidden_size),
            },
            "hy": {
                "kernel": jax.random.normal(key_hy, (output_size, hidden_size)),
                "bias": np.zeros(output_size),
            },
        }

    def forward_train(self, params, x, h0):
        """ Forward pass, apply the network recurrently for the train sequence length.

            x.shape = (batch_size, self.train_seq_length, input_size)
            h0.shape = (batch_size, hidden_size)

            Returns also intermediate values to implement backpropagation.
        """

        batch_size = x.shape[0]

        # store the intermediate values (before the activation function)
        zs = {
            "z_h": [],
            "z_x": [],
            "z_y": [],
        }

        h = h0
        hs = []
        hs.append(h)

        for t in range(self.train_seq_length):

            z_x = (params["hhx"]["kernel_x"][None, :, :]
                   * x[:, t, None, :]).sum(axis=-1)

            z_h = (params["hhx"]["kernel_h"][None, :, :]
                   * h[:, None, :]).sum(axis=-1)

            h = sigmoid(z_x + z_h + params["hhx"]["bias"])

            z_y = (params["hy"]["kernel"][None, :, :]
                   * h[:, None, :]).sum(axis=-1)

            y = sigmoid(z_y + params["hy"]["bias"])
