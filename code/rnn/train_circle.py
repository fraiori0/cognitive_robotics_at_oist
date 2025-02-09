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
n_batch_samples = 32
learning_rate = 0.0002 * n_batch_samples

data_name = "two_circles_opp"

"""------------------"""
""" Model """
"""------------------"""

input_size = 2
hidden_size = 16
output_size = 2

train_seq_length = 150

h0_min = 0.3  # -0.8  #
h0_max = 0.7

hidden_activation = "sigmoid"
output_activation = "sigmoid"

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
}

SEED_PARAMS = 1

rnn = RNN(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    train_seq_length=train_seq_length,
    hidden_activation_fn=activation_dict[hidden_activation]["fn"],
    hidden_activation_fn_grad=activation_dict[hidden_activation]["grad"],
    output_activation_fn=activation_dict[output_activation]["fn"],
    output_activation_fn_grad=activation_dict[output_activation]["grad"],
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
    data_name,
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
    "h0_min": h0_min,
    "h0_max": h0_max,
    "hidden_activation": hidden_activation,
    "output_activation": output_activation,
}

name_suffix = f"_{data_name}_{hidden_size}"
model_name = "rnn" + name_suffix

# try to load the model's parameters, if they exist
if os.path.exists(os.path.join(model_path, f"{model_name}_params.pkl")):
    with open(os.path.join(model_path, f"{model_name}_params.pkl"), "rb") as f:
        rnn.params = pickle.load(f)
    # with open(os.path.join(model_path, f"{model_name}_info.json"), "r") as f:
    #     model_info_dict = json.load(f)


"""------------------"""
""" Data """
"""------------------"""

N_STEPS_MAX = 1000

print("\nImporting Data")

data_folder = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    os.pardir,
    os.pardir,
    "data",
)

# load the data, all file in the targe folder with the name {data_name}_N*.json, where N is an integer
data_files = [f for f in os.listdir(
    data_folder) if f.startswith(f"{data_name}_")]

# import the data
data = {}
for f in data_files:
    with open(os.path.join(data_folder, f), "r") as file:
        # remove the file extension
        k = f.rsplit(".", 1)[0]
        data[k] = json.load(file)

print(f"\tData imported, found >> {len(data)} << files")

ps_dict = {k: np.array(data[k]["p"]) for k in data.keys()}

# print minimum and maximum of each component of each trajectory
for k in ps_dict.keys():
    print(f"\t{k}")
    print(
        f"\t\tX: min {ps_dict[k][:,0].min(axis=0)}, max {ps_dict[k][:,0].max(axis=0)}")
    print(
        f"\t\tY: min {ps_dict[k][:,1].min(axis=0)}, max{ps_dict[k][:,1].max(axis=0)}")


# for each of the data files, extract the sequences and create a moving window
# version of them, which will provide the small training sequences
print(f"\tApplying moving window to the data")

for k in ps_dict.keys():

    # for each of the data files, extract the sequences and create a moving window
    # version of them, which will provide the small training sequences
    ps = ps_dict[k]

    # cut the data to the maximum number of steps, taking the central part
    tot_steps = ps.shape[0]
    i_start = (tot_steps - N_STEPS_MAX) // 2
    i_start = max(0, i_start)
    i_stop = i_start + N_STEPS_MAX
    i_stop = min(tot_steps, i_stop)
    ps = ps[i_start:i_stop]
    n_steps = ps.shape[0]

    # apply moving window
    idx = np.arange(
        n_steps - train_seq_length)[:, None] + np.arange(train_seq_length)
    ps_dict[k] = {
        "x": ps[idx],
        "y": ps[idx + 1],
        # add also a number to identify the trajectory file
        "id": int(k.split("_")[-1]),
    }

# print the length of the sequences for every file
print(f"\tData processed, the following training sequences are available:")
for k in ps_dict.keys():
    print(f"\t\t{k}: {ps_dict[k]['x'].shape}")
    print(f"\t\t\tid: {ps_dict[k]['id']}")


"""------------------"""
""" Test the forward and backprop """
"""------------------"""

# test the forward pass
print("\nTesting forward_training pass")

xs = ps_dict[f"{data_name}_0"]["x"][:n_batch_samples]
ys_target = ps_dict[f"{data_name}_0"]["y"][:n_batch_samples]
# initial hidden state
key = jax.random.key(0)
h0 = rnn.gen_hidden_state_uniform(key, n_batch_samples, h0_min, h0_max)
forward_train_jit = jit(rnn.forward_train)
# forward pass
zs = forward_train_jit(rnn.params, xs, h0)

print(f"\tShape of the forward pass")

print(f"\t\txs, {xs.shape}")
for k in zs.keys():
    print(f"\t\t{k}, {zs[k].shape}")
    # print also if there is any nan or inf
    if np.isnan(zs[k]).any() or np.isinf(zs[k]).any():
        raise ValueError(f"\t\t{k}, nan or inf")


# test backpropagation
print("\nTesting backpropagation")

backprop_jit = jit(rnn.backprop)
w_grad = backprop_jit(rnn.params, zs, xs, ys_target)

print(f"\tShape of gradients")
for k in w_grad.keys():
    for kk in w_grad[k].keys():
        print(f"\t\t{k}, {kk}, {w_grad[k][kk].shape}")
    # print also if there is any nan or inf
    if np.isnan(w_grad[k][kk]).any() or np.isinf(w_grad[k][kk]).any():
        raise ValueError(f"\t\t{k}, nan or inf")

# test params update
print("\nTesting params update")

update_weights_jit = jit(rnn.update_weights)
new_params = update_weights_jit(rnn.params, w_grad, learning_rate)

print(f"\tShape of params after update")
for k in new_params.keys():
    for kk in new_params[k].keys():
        print(f"\t\t{k}, {kk}, {new_params[k][kk].shape}")
    # print also if there is any nan or inf
    if np.isnan(new_params[k][kk]).any() or np.isinf(new_params[k][kk]).any():
        raise ValueError(f"\t\t{k}, nan or inf")

# concatenate the mini-sequences
X = []
Y = []
IDS = []
for k in ps_dict.keys():
    X.append(ps_dict[k]["x"])
    Y.append(ps_dict[k]["y"])
    IDS.append(np.ones(ps_dict[k]["x"].shape[0], dtype=int) * ps_dict[k]["id"])


X = np.concatenate(X, axis=0)
Y = np.concatenate(Y, axis=0)
IDS = np.concatenate(IDS, axis=0)

print(f"\nData concatenated, shape of whole dataset:")
print(f"\tX: {X.shape}")
print(f"\tY: {Y.shape}")
print(f"\tIDS: {IDS.shape}")


"""------------------"""
""" Training """
"""------------------"""

try:
    for ne in range(n_epochs):

        training_sequence = f"{data_name}_0"

        print(f"\nEpoch {ne}")

        # shuffle the sequences
        key, _ = jax.random.split(key)
        idx = jax.random.permutation(key, np.arange(X.shape[0]))

        X = X[idx]
        Y = Y[idx]

        # iterate over the training sequences in mini-batches
        for nb, i_start in enumerate(range(0, X.shape[0], n_batch_samples)):

            # print(f"\tBatch {nb}")

            # print(i_start)

            i_stop = min(i_start + n_batch_samples, X.shape[0])

            xs = X[i_start:i_stop]
            ys_target = Y[i_start:i_stop]

            # print(xs.shape)
            # print(ys_target.shape)

            # exit()

            # initial hidden state
            key, _ = jax.random.split(key)
            h0 = rnn.gen_hidden_state_uniform(key, xs.shape[0], h0_min, h0_max)

            # forward pass
            zs = forward_train_jit(rnn.params, xs, h0)

            # # check for nan
            # for k in zs.keys():
            #     if np.isnan(zs[k]).any() or np.isinf(zs[k]).any():
            #         print(zs["z_hx"])
            #         print(zs["z_hx"].mean())
            #         print(zs["z_hx"].max())
            #         print(zs["z_hx"].min())
            #         raise ValueError(f"\t\t{k}, nan or inf")

            # backpropagation
            w_grad = backprop_jit(rnn.params, zs, xs, ys_target)

            # # print the maximum and minimum of all gradient for each key
            # for k in w_grad.keys():
            #     for kk in w_grad[k].keys():
            #         print(
            #             f"\t\t{k}, {kk}, {w_grad[k][kk].max()}, {w_grad[k][kk].min()}"
            #         )

            # update the weights
            rnn.params = update_weights_jit(rnn.params, w_grad, learning_rate)

        # compute and print mean loss over all sequences

        # initial hidden state
        key, _ = jax.random.split(key)
        h0 = rnn.gen_hidden_state_uniform(key, X.shape[0], h0_min, h0_max)

        zs = forward_train_jit(rnn.params, X, h0)

        l = loss(Y, zs["y"])

        print(f"\tLoss: {l[:, 3:].mean()}")

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
