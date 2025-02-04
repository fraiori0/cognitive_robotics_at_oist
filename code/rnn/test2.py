import os
import jax
import jax.numpy as np
from jax import grad, jit, vmap
from backprop import *
from time import time
from functools import partial
import plotly.graph_objects as go
import json
import pickle

os.environ["JAX_TRACEBACK_FILTERING"] = "off"

SAVE = True

n_epochs = 1000
n_batch_samples = 10
learning_rate = 0.01


"""------------------"""
""" Model """
"""------------------"""

input_size = 2
hidden_size = 2
output_size = 2

train_seq_length = 20

SEED_PARAMS = 0

rnn = RNN(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    train_seq_length=train_seq_length,
    seed=SEED_PARAMS,
)

for k in rnn.params.keys():
    for kk in rnn.params[k].keys():
        print(f"{k}, {kk}, {rnn.params[k][kk].shape}")


# path to the model's folder
model_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    os.pardir,
    os.pardir,
    "models",
)

# create the folder if it does not exist
if not os.path.exists(model_path):
    os.makedirs(model_path)

# dictionary with the model's information
model_info_dict = {
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "train_seq_length": train_seq_length,
    "SEED_PARAMS": SEED_PARAMS,
}

name_suffix = f"_{input_size}_{hidden_size}_{output_size}"
model_name = "rnn" + name_suffix


"""------------------"""
""" Data """
"""------------------"""

N_STEPS = 200

# for now, generate a bunch of data analytically
# later we will change to import from csv

# radius, center, angular speed
r = 0.3
c = np.array((0.5, 0.5))
# about 30 steps per revolution
omega = 2 * np.pi / 30

state = {"r": r, "c": c, "omega": omega}
means = {"r": r, "c": c, "omega": omega}
sigmas = {"r": 0.01, "c": 0.01, "omega": 0.1}
gains = {"r": 0.1, "c": 0.1, "omega": 0.3}


def df(keys_dict, state, means, sigmas, gains):
    # state, state, means and sigmas are dictionary of means and sigmas for each feature

    # compute a delta for the radius, center, and angular speed
    # using a mean-reverting stochastic process, so we stay close to the desired values
    # but also have some noise
    # use the Ornstein-Uhlenbeck process because it is simple

    dstate = {
        "r": gains["r"] * (means["r"] - state["r"])
        + sigmas["r"] * jax.random.normal(keys_dict["r"], (1,))[0],
        "c": gains["c"] * (means["c"] - state["c"])
        + sigmas["c"] * jax.random.normal(keys_dict["c"], (2,)),
        "omega": gains["omega"] * (means["omega"] - state["omega"])
        + sigmas["omega"] * jax.random.normal(keys_dict["omega"], (1,))[0],
    }

    return dstate


df_jit = jit(partial(df, means=means, sigmas=sigmas, gains=gains))


def update_state(keys_dict, state, means, sigmas, gains):
    # update the state using the delta
    dstate = df(keys_dict, state, means, sigmas, gains)
    state = {k: state[k] + dstate[k] for k in state.keys()}
    # clip omega to not be too low or too high
    state["omega"] = np.clip(state["omega"], 0.1, 0.5)
    return state


update_state_jit = jit(partial(update_state, means=means, sigmas=sigmas, gains=gains))


def gen_seq(key, state, length=10):
    key, _ = jax.random.split(key)
    # start from a random angular position
    theta = jax.random.uniform(key, (1,), minval=-np.pi, maxval=np.pi)

    seq_p = []
    seq_theta = []

    for _ in range(length):
        # compute the new angular position
        theta = theta + state["omega"]
        # compute the position from the angular position
        p = state["c"] + state["r"] * np.array([np.cos(theta), np.sin(theta)]).squeeze()

        seq_p.append(p)
        seq_theta.append(theta)

        key, *keys = jax.random.split(key, 4)
        keys_dict = {k: keys[i] for i, k in enumerate(state.keys())}

        # update the state
        state = update_state_jit(keys_dict, state)

    return np.array(seq_p), np.array(seq_theta)


# generate a sequence
key = jax.random.PRNGKey(0)

t0 = time()
seq_p, seq_theta = gen_seq(key, state, length=N_STEPS)
print("Time to generate data:", time() - t0)

# convert data to training sequences
idx = np.arange(N_STEPS - train_seq_length)[:, None] + np.arange(train_seq_length)
X = seq_p[idx]
Y = seq_p[idx + 1]


"""------------------"""
""" Test the forward and backprop """
"""------------------"""

# test the forward pass
xs = X[:n_batch_samples]
ys_target = Y[:n_batch_samples]

# initial hidden state
key, _ = jax.random.split(key)
h0 = rnn.gen_hidden_state(key, n_batch_samples)

forward_train_jit = jit(rnn.forward_train)

# forward pass
zs = forward_train_jit(rnn.params, xs, h0)

print(f"xs, {xs.shape}")
for k in zs.keys():
    print(f"{k}, {zs[k].shape}")


# test backpropagation

backprop_jit = jit(rnn.backprop)

# initial hidden state
w_grad = backprop_jit(rnn.params, zs, xs, ys_target)


for k in w_grad.keys():
    for kk in w_grad[k].keys():
        print(f"{k}, {kk}, {w_grad[k][kk].shape}")

# test params update
update_weights_jit = jit(rnn.update_weights)

new_params = update_weights_jit(rnn.params, w_grad, learning_rate)

for k in new_params.keys():
    for kk in new_params[k].keys():
        print(f"{k}, {kk}, {new_params[k][kk].shape}")

# exit()


"""------------------"""
""" Training """
"""------------------"""

try:
    for ne in range(n_epochs):

        print(f"\nEpoch {ne}")

        # shuffle the sequences
        key, _ = jax.random.split(key)
        idx = jax.random.permutation(key, np.arange(X.shape[0]))

        X = X[idx]
        Y = Y[idx]

        # iterate over the training sequences in mini-batches
        for i_start in range(0, X.shape[0], n_batch_samples):

            # print(i_start)

            i_stop = min(i_start + n_batch_samples, X.shape[0])

            xs = X[i_start:i_stop]
            ys_target = Y[i_start:i_stop]

            # print(xs.shape)
            # print(ys_target.shape)

            # exit()

            # initial hidden state
            key, _ = jax.random.split(key)
            h0 = rnn.gen_hidden_state(key, xs.shape[0])

            # forward pass
            zs = forward_train_jit(rnn.params, xs, h0)

            # backpropagation
            w_grad = backprop_jit(rnn.params, zs, xs, ys_target)

            # update the weights
            rnn.params = update_weights_jit(rnn.params, w_grad, learning_rate)

        # compute and print mean loss over all sequences

        # initial hidden state
        key, _ = jax.random.split(key)
        h0 = rnn.gen_hidden_state(key, X.shape[0])

        zs = forward_train_jit(rnn.params, X, h0)

        l = loss(Y, zs["y"])

        print(f"\tLoss: {l[:, 5:].mean()}")

        # save the parameters every 5 epochs
        if SAVE and (ne % 5 == 0) and (ne > 0):
            with open(os.path.join(model_path, f"{model_name}_info.json"), "w") as f:
                json.dump(model_info_dict, f)
            with open(os.path.join(model_path, f"{model_name}_params.pkl"), "wb") as f:
                pickle.dump(rnn.params, f)

except KeyboardInterrupt:
    pass


# save the parameters
if SAVE:
    with open(os.path.join(model_path, f"{model_name}_info.json"), "w") as f:
        json.dump(model_info_dict, f)
    with open(os.path.join(model_path, f"{model_name}_params.pkl"), "wb") as f:
        pickle.dump(rnn.params, f)
