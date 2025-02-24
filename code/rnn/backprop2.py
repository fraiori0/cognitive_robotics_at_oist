import jax
import jax.numpy as np
from jax import grad, jit, vmap
from functools import partial


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


# ReLU
def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return np.where(x > 0, 1, 0)


# Leaky ReLU
def leaky_relu(x):
    return np.maximum(0.01 * x, x)


def leaky_relu_grad(x):
    return np.where(x > 0, 1, 0.01)


activation_dict = {
    "sigmoid": {
        "fn": sigmoid,
        "grad": sigmoid_grad,
    },
    "tanh": {
        "fn": tanh,
        "grad": tanh_grad,
    },
    "sin": {
        "fn": sin,
        "grad": sin_grad,
    },
    "relu": {
        "fn": relu,
        "grad": relu_grad,
    },
    "leaky_relu": {
        "fn": leaky_relu,
        "grad": leaky_relu_grad,
    },
}


# loss function 1/2 squared error
def loss(y_true, y_pred):
    return (0.5 * (y_true - y_pred) ** 2).sum(axis=-1)


# derivative of the squared error loss function
def loss_grad(y_true, y_pred):
    return y_pred - y_true


def apply_grads(params, grads, learning_rate, train_seq_length):
    return params - learning_rate * grads.sum(axis=0) / train_seq_length


def add_values(dict1, dict2):
    return dict1 + dict2


class RNN:
    """Vanilla Recurrent Network with Sigmoid activation function."""

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        hidden_activation_fn: str = "sigmoid",
        output_activation_fn: str = "sigmoid",
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
        self.hidden_activation_fn = activation_dict[hidden_activation_fn]["fn"]
        self.hidden_activation_fn_grad = activation_dict[hidden_activation_fn]["grad"]
        self.output_activation_fn = activation_dict[output_activation_fn]["fn"]
        self.output_activation_fn_grad = activation_dict[output_activation_fn]["grad"]

        # random key
        self.key = jax.random.key(seed)

        # generate the weights by sampling from the normal distribution
        # and clipping, like normal_lecun in flax.linen
        key_h, key_x, key_y = jax.random.split(self.key, 3)

        self.params = {
            "hidden": {
                "Wx": np.clip(
                    jax.random.normal(key_x, (hidden_size, input_size)), -1, 1
                ),
                "Wh": np.clip(
                    jax.random.normal(key_h, (hidden_size, hidden_size)), -1, 1
                ),
                "bh": np.zeros(hidden_size),
            },
            "output": {
                "Wy": np.clip(
                    jax.random.normal(key_y, (output_size, hidden_size)), -1, 1
                ),
                "by": np.zeros(output_size),
            },
        }

        self.forward_train_single_jit = jit(self.forward_train_single)
        self.backprop_single_jit = jit(self.backprop_single)

    def forward_train_single(self, params, x, h0):
        """Forward pass for a single time step
        Returns also intermdiate values needed for backpropagation.
        Function useful for JIT compilation.

        NOTE: here we assume a single time-step, and we are missing the time dimension in the input x
        x.shape = (batch_size, input_size)
        h0.shape = (batch_size, hidden_size)
        """

        zs = {
            # previous hidden state
            "h_prev": h0,
            # input
            "x": x,
            # intermediate value hidden
            "z_h": None,
            # hidden state
            "h": None,
            # intermediate value output
            "z_y": None,
            # output
            "y": None,
        }

        z_h = (
            (params["hidden"]["Wx"] * x[:, None, :]).sum(axis=-1)
            + (params["hidden"]["Wh"] * h0[:, None, :]).sum(axis=-1)
            + params["hidden"]["bh"]
        )
        h = self.hidden_activation_fn(z_h)

        z_y = (params["output"]["Wy"] * h[:, None, :]).sum(axis=-1) + params["output"][
            "by"
        ]
        y = self.output_activation_fn(z_y)

        zs["z_h"] = z_h
        zs["h"] = h
        zs["z_y"] = z_y
        zs["y"] = y

        return zs

    def forward_train(self, key, params, x, h0):
        """Forward pass, apply the network recurrently for the train_seq_length time steps.
        Returns also intermdiate values needed for backpropagation.
        The number of steps is fixed in order to be JIT-compilable;
            do not exagerate with time steps otherwise will take long to compile, it's a for cycle.


        NOTE: here we assume time axis as the second-to-last dimension in the input x
            no time dimension on h0 because intermediate states will be generated.

        x.shape = (batch_size, self.train_seq_length, input_size)
        h0.shape = (batch_size, hidden_size)

        NOTE 2
        the input ratio control mask is 1 if we used the predicted value
        from the previous step as input (i.e., x(t)=y(t-1)),
        and is 0 if we used the true value of x(t) passed in the input to this function.
        """

        # values needed for backpropagation
        zs = {
            # previous hidden state
            "h_prev": [],
            # input
            "x": [],
            # input ratio control mask
            "irc_mask": [],
            # intermediate values hidden
            "z_h": [],
            # hidden state, NOTE, has one extra element
            "h": [],
            # intermediate value output
            "z_y": [],
            # output
            "y": [],
        }

        # perform a first forward pass, after that we may apply input ratio control
        # (i.e., the first input x cannot be self-generated)
        zs_step = self.forward_train_single_jit(params, x[:, 0, :], h0)
        zs_step["irc_mask"] = np.zeros((x.shape[0],))
        # append value maunually, no 'for' cycle to improve jitting
        zs["h_prev"].append(zs_step["h_prev"])
        zs["x"].append(zs_step["x"])
        zs["irc_mask"].append(zs_step["irc_mask"])
        zs["z_h"].append(zs_step["z_h"])
        zs["h"].append(zs_step["h"])
        zs["z_y"].append(zs_step["z_y"])
        zs["y"].append(zs_step["y"])

        # loop forward in time
        # NOTE! STARTING FROM 1, the first input was already processed
        for t in range(1, self.train_seq_length):

            # decide whether to use the predicted value or the true value
            key, _ = jax.random.split(key)
            # NOTE: self.p_input_ratio is the probability of using the true value
            # so we use (1-self.p_input_ratio) to decide whether to use the predicted value
            mask = jax.random.bernoulli(
                key, 1.0 - self.p_input_ratio, (x.shape[0])
            ).astype(np.float32)

            # if mask is 1 we use predicted, otherwise true value
            x_input = (
                x[:, t, :] * (1 - mask)[..., None] + zs_step["y"] * mask[..., None]
            )

            # forward pass
            zs_step = self.forward_train_single_jit(params, x_input, zs_step["h"])
            zs_step["irc_mask"] = mask

            # append value maunually, no 'for' cycle to improve jitting
            zs["h_prev"].append(zs_step["h_prev"])
            zs["x"].append(zs_step["x"])
            zs["irc_mask"].append(zs_step["irc_mask"])
            zs["z_h"].append(zs_step["z_h"])
            zs["h"].append(zs_step["h"])
            zs["z_y"].append(zs_step["z_y"])
            zs["y"].append(zs_step["y"])

            # zs["irc_mask"].append(mask)

        # stack values manually, no 'for' cycle to improve jitting
        zs["h_prev"] = np.stack(zs["h_prev"], axis=1)
        zs["x"] = np.stack(zs["x"], axis=1)
        zs["irc_mask"] = np.stack(zs["irc_mask"], axis=1)
        zs["z_h"] = np.stack(zs["z_h"], axis=1)
        zs["h"] = np.stack(zs["h"], axis=1)
        zs["z_y"] = np.stack(zs["z_y"], axis=1)
        zs["y"] = np.stack(zs["y"], axis=1)

        return zs

    def forward(self, params, x, h):
        """Forward pass, apply the network for one time step.
        Does not return intermediate value, useful after training.

        x.shape = (batch_size, input_size)
        h.shape = (batch_size, hidden_size)

        """
        z_h = (
            (params["hidden"]["Wx"] * x[:, None, :]).sum(axis=-1)
            + (params["hidden"]["Wh"] * h[:, None, :]).sum(axis=-1)
            + params["hidden"]["bh"]
        )
        h = self.hidden_activation_fn(z_h)

        z_y = (params["output"]["Wy"] * h[:, None, :]).sum(axis=-1) + params["output"][
            "by"
        ]
        y = self.output_activation_fn(z_y)

        return y, h

    def backprop_single(
        self,
        params,
        zs,
        y_true,
        delta_h1,
        grad_h1_h,
        grad_h1_y,
    ):
        """Backward pass for a single time step, to be JIT-compiled.

        zs = as returned from the full  when passing x of (batch_size, input_size)
        y_true.shape = (batch_size, output_size)

        NOTE 2: we give as input also the 'carry' values for the error and gradients
            on the next (in time) and previous (in the the cycle) hidden state
        """

        # update the gradients of the parameters using the above values
        # gradient of loss at time t w.r.t. y(t), shape (batch_size, output_size)
        grad_loss_t_y_t = loss_grad(y_true, zs["y"])
        # gradient of y(t) with respect to h(t) shape (batch_size, output_size, hidden_size)
        grad_y_h = params["output"]["Wy"] * self.output_activation_fn_grad(
            zs["z_y"][..., :, None]
        )

        # hidden state error shape (batch_size, hidden_size)
        delta_h1 = (grad_loss_t_y_t[..., None] * grad_y_h).sum(axis=-2) + (
            delta_h1[..., None] * grad_h1_h
        ).sum(axis=-2)
        # output error shape (batch_size, output_size)
        delta_y_t = grad_loss_t_y_t + (delta_h1[..., None] * grad_h1_y).sum(axis=-2)

        # gradient of y(t) w.r.t. Wy shape (batch_size, output_size, hidden_size)
        # TODO: check if this is correct
        grad_y_Wy = zs["h"][..., None, :] * self.output_activation_fn_grad(
            zs["z_y"][..., :, None]
        )
        # gradient of y(t) w.r.t. by shape (batch_size, output_size)
        grad_y_by = self.output_activation_fn_grad(zs["z_y"][..., :])
        # gradient of h(t) w.r.t. Wh shape (batch_size, hidden_size, hidden_size)
        grad_h_Wh = zs["h_prev"][..., None, :] * self.hidden_activation_fn_grad(
            zs["z_h"][..., :, None]
        )
        # gradient of h(t) w.r.t. Wx shape (batch_size, hidden_size, input_size)
        grad_h_Wx = zs["x"][..., None, :] * self.hidden_activation_fn_grad(
            zs["z_h"][..., :, None]
        )
        # gradient of h(t) w.r.t. bh shape (batch_size, hidden_size)
        grad_h_bh = self.hidden_activation_fn_grad(zs["z_h"][..., :])

        # update the gradients of the parameters using the above values
        grads_t = {
            "hidden": {},
            "output": {},
        }
        grads_t["output"]["Wy"] = delta_y_t[..., None] * grad_y_Wy
        grads_t["output"]["by"] = delta_y_t * grad_y_by
        grads_t["hidden"]["Wh"] = delta_h1[..., None] * grad_h_Wh
        grads_t["hidden"]["Wx"] = delta_h1[..., None] * grad_h_Wx
        grads_t["hidden"]["bh"] = delta_h1 * grad_h_bh

        # # # Update gradients of h(t+1), here we use the mask
        # gradient of h(t+1) w.r.t. h(t) shape (batch_size, hidden_size, hidden_size)
        grad_h1_h = self.hidden_activation_fn_grad(zs["z_h"][..., None, :]) * (
            # (hidden, hidden)
            params["hidden"]["Wh"]
            # (batch, None, None)
            + zs["irc_mask"][:, None, None]
            # (batch, hidden, hidden)
            # NOTE, here is important that input_size = output_size
            * (
                # (hidden, input, None)
                params["hidden"]["Wx"][..., None]
                # (batch, None, output, None)
                * self.output_activation_fn_grad(zs["z_y"][..., None, :, None])
                # (None, output, hidden)
                * params["output"]["Wy"][None]
            ).sum(axis=-2)
        )

        # gradient of h(t+1) w.r.t. y(t) shape (batch_size, hidden_size, output_size)
        grad_h1_y = (
            # (batch, None, None)
            zs["irc_mask"][:, None, None]
            # (batch, hidden, None)
            * self.hidden_activation_fn(zs["z_h"][..., :, None])
            # (hidden, input)
            * params["hidden"]["Wx"]
        )

        return grads_t, delta_h1, grad_h1_h, grad_h1_y

    def backprop(self, params, zs, y_true):
        """Backward pass, compute the gradients of the loss with respect to the parameters.

        zs = as returned from self.forward_train when passing x of (batch_size, self.train_seq_length, output_size)
        y_true.shape = (batch_size, self.train_seq_length, output_size)
        """

        batch_dims = y_true.shape[:-2]

        # initialize the gradients, same shape as the parameters with added batch dimensions
        # in case we want to keep them separate in the batch
        grads = {
            "hidden": {
                "Wx": np.zeros((*batch_dims, *params["hidden"]["Wx"].shape)),
                "Wh": np.zeros((*batch_dims, *params["hidden"]["Wh"].shape)),
                "bh": np.zeros((*batch_dims, *params["hidden"]["bh"].shape)),
            },
            "output": {
                "Wy": np.zeros((*batch_dims, *params["output"]["Wy"].shape)),
                "by": np.zeros((*batch_dims, *params["output"]["by"].shape)),
            },
        }

        # # Initialize the hidden state error
        delta_h1 = np.zeros((*batch_dims, self.hidden_size))
        # # And gradients of next hidden state
        # gradient of h(t+1) w.r.t. h(t)
        grad_h1_h = np.zeros((*batch_dims, self.hidden_size, self.hidden_size))
        # gradient of h(t+1) w.r.t. y(t)
        grad_h1_y = np.zeros((*batch_dims, self.hidden_size, self.output_size))

        # iterate from the last time step to the first
        for t in reversed(range(self.train_seq_length)):

            zs_step = jax.tree_util.tree_map(
                partial(
                    np.take,
                    indices=t,
                    axis=1,
                ),
                zs,
            )

            # compute the gradients for the time step t
            grads_t, delta_h1, grad_h1_h, grad_h1_y = self.backprop_single_jit(
                params,
                zs_step,
                y_true[:, t, :],
                delta_h1,
                grad_h1_h,
                grad_h1_y,
            )

            # update the gradients
            grads = jax.tree_util.tree_map(add_values, grads, grads_t)

        # compute the gradient for the initial hidden state
        # which is the hidden state error for the first state
        grad_loss_t_y_t = loss_grad(y_true[:, 0], zs["y"][:, 0])
        # gradient of y(t) with respect to h(t) shape (batch_size, output_size, hidden_size)
        grad_y_h = params["output"]["Wy"] * self.output_activation_fn_grad(
            zs["z_y"][..., 0, :, None]
        )

        grads["initial"] = {
            "h0": (grad_loss_t_y_t[..., None] * grad_y_h).sum(axis=-2)
            + (delta_h1[..., None] * grad_h1_h).sum(axis=-2)
        }

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
        grads = grads.copy()
        grads.pop("initial")
        params = jax.tree_util.tree_map(
            partial(
                apply_grads,
                learning_rate=learning_rate,
                train_seq_length=self.train_seq_length,
            ),
            params,
            grads,
        )

        # for k in params.keys():
        #     for kk in params[k].keys():
        #         params[k][kk] = (
        #             params[k][kk]
        #             - learning_rate
        #             # NOTE: we divide by the train_seq_length
        #             # because in backprop we sum, in this way it become like a mean
        #             # of the gradients for the loss at each time step
        #             # and we have similar update magnitude for different train_seq_length
        #             * (grads[k][kk]).sum(axis=0) / self.train_seq_length
        #         )

        return params
