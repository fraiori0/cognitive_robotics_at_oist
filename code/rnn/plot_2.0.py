import os
import jax
import jax.numpy as np
from jax import grad, jit, vmap
import numpy as onp
from backprop2 import *
from time import time
from functools import partial
import plotly.graph_objects as go
import json
import pickle
from sklearn.decomposition import PCA

os.environ["JAX_TRACEBACK_FILTERING"] = "off"

SAVE = True

data_name = "one_circle"
test_name = "v2_rh0"
hidden_size = 2

SEED = 3
key = jax.random.key(SEED)
n_steps_rollout = 50

# random_hidden_states = True

initial_seq_length = 5
moving_window_step = 19
n_gradient_updates_hidden = 1000
learning_rate_hidden = 0.001

"""------------------"""
""" Model """
"""------------------"""

# path to the model's folder
model_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    os.pardir,
    os.pardir,
    "models",
    data_name,
)

name_suffix = f"_{data_name}_{hidden_size}_{test_name}"
model_name = "rnn" + name_suffix

# import the model's parameters if they exist
if os.path.exists(os.path.join(model_path, f"{model_name}_params.pkl")):
    with open(os.path.join(model_path, f"{model_name}_params.pkl"), "rb") as f:
        params = pickle.load(f)
    with open(os.path.join(model_path, f"{model_name}_info.json"), "r") as f:
        model_info_dict = json.load(f)
    # and also the initial hidden states
    with open(os.path.join(model_path, f"{model_name}_h0.pkl"), "rb") as f:
        hidden_states = pickle.load(f)
        # if there was only one trajectory, add a leading dimension to the hidden states
        for k in hidden_states.keys():
            if len(hidden_states[k]["h0"].shape) == 1:
                hidden_states[k]["h0"] = hidden_states[k]["h0"][None, :]
    # and also the training history
    with open(os.path.join(model_path, f"{model_name}_history.pkl"), "rb") as f:
        history = pickle.load(f)
else:
    print("Could not load parameters")
    raise ValueError("Could not load parameters")

SEED_PARAMS = 0

rnn = RNN(
    input_size=model_info_dict["input_size"],
    hidden_size=hidden_size,
    output_size=model_info_dict["output_size"],
    train_seq_length=initial_seq_length,
    hidden_activation_fn=model_info_dict["hidden_activation"],
    output_activation_fn=model_info_dict["output_activation"],
    seed=SEED_PARAMS,
    p_input_ratio=1.0,
)

# Set parameters
rnn.params = params


print(f"\nModel loaded, {model_name}")

print(f"\nParams shape:")
for k in rnn.params.keys():
    for kk in rnn.params[k].keys():
        print(f"\t{k}, {kk}, {rnn.params[k][kk].shape}")

print(f"\nInitial hidden states:")
for k in hidden_states.keys():
    print(f"\t{k}, ID: {hidden_states[k]['id']}")
    print(f"\t\tH0: {hidden_states[k]['h0'].shape}")


# exit()

"""------------------"""
""" Import and Process Data """
"""------------------"""

# sequences longer than this will be cut, preserving the central part
# can be useful to remove initial and final parts of the trajectories
N_STEPS_MAX = 1000

print("\nImporting Data")

data_folder = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    os.pardir,
    os.pardir,
    "data",
)

# load the data, all file in the targe folder with the name {data_name}_N*.json, where N is an integer
data_files = [f for f in os.listdir(data_folder) if f.startswith(f"{data_name}_")]

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
        f"\t\tX: min {ps_dict[k][:,0].min(axis=0)}, max {ps_dict[k][:,0].max(axis=0)}"
    )
    print(f"\t\tY: min {ps_dict[k][:,1].min(axis=0)}, max{ps_dict[k][:,1].max(axis=0)}")


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
    n_step_seq = ps.shape[0]

    # apply moving window
    idx = np.arange(
        start=0,
        stop=n_step_seq - initial_seq_length,
        step=moving_window_step,
        dtype=np.int32,
    )[:, None]
    idx = idx + np.arange(initial_seq_length, dtype=np.int32)
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

# exit()

"""------------------"""
""" Create single dataset """
"""------------------"""

# concatenate the mini-sequences into a single batch dimension
# creating a single dataset
X = []
Y = []
IDS = []
# this array will not be shuffled, it's indexes correspond to the hidden_states for
# trajectories of such ID
H0_map = []
for k in ps_dict.keys():
    X.append(ps_dict[k]["x"])
    Y.append(ps_dict[k]["y"])
    IDS.append(np.ones(ps_dict[k]["x"].shape[0], dtype=int) * ps_dict[k]["id"])
    H0_map.append(hidden_states[k]["h0"])


X = np.concatenate(X, axis=0)
Y = np.concatenate(Y, axis=0)
IDS = np.concatenate(IDS, axis=0)
H0_map = np.concatenate(H0_map, axis=0)


print(f"\nData concatenated, shape of whole dataset:")
print(f"\tX: {X.shape}")
print(f"\tY: {Y.shape}")
print(f"\tIDS: {IDS.shape}")
print(f"\tH0_map: {H0_map.shape}")

"""------------------"""
""" Jit Forward and Backward """
"""------------------"""

forward_jit = jit(rnn.forward)

# with JIT is much much faster, but intense to compile depending on how many time-step we have
# in the forward pass
forward_train_jit = jit(rnn.forward_train)  # rnn.forward_train  #
# forward pass
key, _ = jax.random.split(key)
zs = forward_train_jit(key, rnn.params, X, H0_map[IDS])

print(f"\tShape of the forward pass")

print(f"\t\txs, {X.shape}")
for k in zs.keys():
    print(f"\t\t{k}, {zs[k].shape}")
    # print also if there is any nan or inf
    if np.isnan(zs[k]).any() or np.isinf(zs[k]).any():
        raise ValueError(f"\t\t{k}, nan or inf")

# test backpropagation
print("\nTesting backpropagation")

# Same here,
# with JIT is much much faster, but intense to compile depending on how many time-step we have
# in the forward pass
backprop_jit = jit(rnn.backprop)
w_grad = backprop_jit(rnn.params, zs, Y)

print(f"\tShape of gradients")
for k in w_grad.keys():
    for kk in w_grad[k].keys():
        print(f"\t\t{k}, {kk}, {w_grad[k][kk].shape}")
    # print also if there is any nan or inf
    if np.isnan(w_grad[k][kk]).any() or np.isinf(w_grad[k][kk]).any():
        raise ValueError(f"\t\t{k}, nan or inf")

# exit()


"""------------------"""
""" Update Initial Hidden States """
"""------------------"""

# Perform forward pass on each of the starting trajectories
# doing a few gradient update on the initial hidden states
h0 = H0_map[IDS]
for i in range(n_gradient_updates_hidden):
    key, _ = jax.random.split(key)
    # forward pass
    zs = forward_train_jit(key, rnn.params, X, h0)
    # backpropagation
    w_grad = backprop_jit(rnn.params, zs, Y)
    # update the hidden states
    h0 = h0 - learning_rate_hidden * w_grad["initial"]["h0"]
    # clip in range
    h0 = np.clip(
        h0, model_info_dict["h0_min_activation"], model_info_dict["h0_max_activation"]
    )


"""------------------"""
""" Generate sequences """
"""------------------"""

# generate sequences
# start from the first x in each sequence
# and from the initial hidden states computed for each sequence
# and iterate in feedback mode for n_steps_rollout steps

x = X[:, 0]
h = h0

# dictionary to store the generated sequences
x_traj_list = [x]
h_traj_list = [h]

# iterate
for ns in range(n_steps_rollout):
    # forward pass
    x, h = forward_jit(rnn.params, x, h)
    # store
    x_traj_list.append(x)
    h_traj_list.append(h)

# convert to arrays
x_traj_list = np.stack(x_traj_list, axis=1)
h_traj_list = np.stack(h_traj_list, axis=1)

print(f"\nGenerated sequences")
print(f"\tx_traj_list: {x_traj_list.shape}")
print(f"\th_traj_list: {h_traj_list.shape}")

# Put into dictionary using the IDS
x_traj_dict = {}
h_traj_dict = {}

for k in ps_dict.keys():
    idx = np.where(IDS == ps_dict[k]["id"])[0]
    x_traj_dict[k] = x_traj_list[idx]
    h_traj_dict[k] = h_traj_list[idx]


"""------------------"""
""" PCA """
"""------------------"""

print(f"\nPerforming PCA")

# perform PCA on the hidden states
pca = PCA(n_components=2)

pca_components_dict = {}

for k in h_traj_dict.keys():
    # fit to (flattened) hidden states
    pca.fit(onp.array(h_traj_dict[k].reshape(-1, h_traj_dict[k].shape[-1])))
    # take the components and convert to JAX array
    # note, it's dimensions are (n_components, n_features)
    pca_components = np.array(pca.components_)
    # store the components
    pca_components_dict[k] = pca_components


# project the (original) hidden states onto the PCA components
# so we keep the dimensions as (n_traj, n_steps, features)

h_traj_pca = {}
for k in h_traj_dict.keys():
    h_traj_pca[k] = (h_traj_dict[k][..., None, :] * pca_components_dict[k]).sum(axis=-1)

# print shape
for k in h_traj_pca.keys():
    print(f"\t{k}")
    print(f"\t  h_traj_pca[{k}]: {h_traj_pca[k].shape}")

# exit()


"""------------------"""
""" Plot Trajectories """
"""------------------"""

# dictionary containing colors for each key (training sequence)
my_colors = [
    "rgba(0, 157, 255, 0.2)",
    "rgba(255, 0, 0, 0.2)",
    "rgba(0, 255, 0, 0.2)",
    "rgba(149, 0, 255, 0.2)",
    "rgba(255, 255, 0, 0.2)",
]

# create a color dictionary
color_dict = {}
for i, k in enumerate(x_traj_dict.keys()):
    color_dict[k] = my_colors[i]


# plot all the trajectories
# in state space (x-y)

fig = go.Figure()

for k in x_traj_dict.keys():
    for i in range(x_traj_dict[k].shape[0]):
        fig.add_trace(
            go.Scatter(
                x=x_traj_dict[k][i, :, 0],
                y=x_traj_dict[k][i, :, 1],
                mode="markers+lines",  # $"markers+lines",  #
                # using a colorscale to represent time,
                # the first point is dark blue, the last is dark red
                marker=dict(
                    color=np.linspace(0, 1, x_traj_dict[k].shape[1]),
                    colorscale="Viridis",
                    colorbar=dict(title="Time"),
                    size=8,
                ),
                # set line color and size
                line=dict(
                    color=color_dict[k],
                    width=2,
                ),
                # legendgroup=k,
            )
        )

fig.update_layout(
    title="Generated data",
    xaxis_title="x",
    yaxis_title="y",
    # set square aspect ratio
    xaxis=dict(scaleanchor="y", scaleratio=1),
    # and axis ranges in 0-1
    # xaxis_range=[0, 1],
    # yaxis_range=[0, 1],
    # set white background and style
    # plot_bgcolor="white",
    template="plotly_white",
    width=800,
    height=800,
)


# add ticks at 0.1 marks, both on x-y axis
fig.update_xaxes(tick0=0, dtick=0.1)
fig.update_yaxes(tick0=0, dtick=0.1)
# set tick color to black


fig.show()

# exit()

"""------------------"""
""" Plot Loss History """
"""------------------"""

# plot the loss history
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=history["loss"],
        mode="lines",
        name="MSE loss",
        line=dict(
            color="rgba(0, 157, 255, 1)",
            width=6,
        ),
    )
)

fig.update_layout(
    title="Loss history",
    xaxis_title="Epoch",
    yaxis_title="Loss",
    # set white background and style
    # plot_bgcolor="white",
    template="plotly_white",
    width=800,
    height=800,
)

fig.show()

# exit()

"""------------------"""
""" Plot PCA """
"""------------------"""

# plot the trajectories of the hidden state as projected on the PCA components
fig = go.Figure()

for k in h_traj_pca.keys():
    for i in range(h_traj_pca[k].shape[0]):
        fig.add_trace(
            go.Scatter(
                x=h_traj_pca[k][i, :, 0],
                y=h_traj_pca[k][i, :, 1],
                mode="markers+lines",  # $"markers+lines",  #
                # using a colorscale to represent time,
                # the first point is dark blue, the last is dark red
                marker=dict(
                    color=np.linspace(0, 1, h_traj_dict[k].shape[1]),
                    colorscale="Viridis",
                    colorbar=dict(title="Time"),
                    size=8,
                ),
                # set line color and size
                line=dict(
                    color=color_dict[k],
                    width=2,
                ),
                # legendgroup=k,
            )
        )

fig.update_layout(
    title="PCA of hidden states",
    xaxis_title="PCA component 1",
    yaxis_title="PCA component 2",
    # set square aspect ratio
    xaxis=dict(scaleanchor="y", scaleratio=1),
    # and axis ranges in 0-1
    # xaxis_range=[-1, 1],
    # yaxis_range=[-1, 1],
    # set white background and style
    # plot_bgcolor="white",
    template="plotly_white",
    width=800,
    height=800,
)

fig.show()

# exit()

"""------------------"""
""" PCA with 3 components """
"""------------------"""

# if there are more than 2 hidden features, perform PCA with 3 components

for k in h_traj_dict.keys():
    if h_traj_dict[k].shape[-1] < 3:
        exit()
print(f"\nPerforming PCA")

# perform PCA on the hidden states
pca = PCA(n_components=3)

pca_components_dict = {}

for k in h_traj_dict.keys():
    # fit to (flattened) hidden states
    pca.fit(onp.array(h_traj_dict[k].reshape(-1, h_traj_dict[k].shape[-1])))
    # take the components and convert to JAX array
    # note, it's dimensions are (n_components, n_features)
    pca_components = np.array(pca.components_)
    # store the components
    pca_components_dict[k] = pca_components


# project the (original) hidden states onto the PCA components
# so we keep the dimensions as (n_traj, n_steps, features)

h_traj_pca = {}
for k in h_traj_dict.keys():
    h_traj_pca[k] = (h_traj_dict[k][..., None, :] * pca_components_dict[k]).sum(axis=-1)

# print shape
for k in h_traj_pca.keys():
    print(f"\t{k}")
    print(f"\t  h_traj_pca[{k}]: {h_traj_pca[k].shape}")

# exit()

# plot the trajectories of the hidden state as projected on the PCA components
# make a 3D plot

fig = go.Figure()

for k in h_traj_pca.keys():
    for i in range(h_traj_pca[k].shape[0]):
        fig.add_trace(
            go.Scatter3d(
                x=h_traj_pca[k][i, :, 0],
                y=h_traj_pca[k][i, :, 1],
                z=h_traj_pca[k][i, :, 2],
                mode="markers+lines",  # $"markers+lines",  #
                # using a colorscale to represent time,
                # the first point is dark blue, the last is dark red
                marker=dict(
                    color=np.linspace(0, 1, h_traj_dict[k].shape[1]),
                    colorscale="Viridis",
                    colorbar=dict(title="Time"),
                    size=8,
                ),
                # set line color and size
                line=dict(
                    color=color_dict[k],
                    width=2,
                ),
                # legendgroup=k,
            )
        )

fig.update_layout(
    title="PCA of hidden states",
    scene=dict(
        xaxis_title="PCA component 1",
        yaxis_title="PCA component 2",
        zaxis_title="PCA component 3",
    ),
    # set white background and style
    # plot_bgcolor="white",
    template="plotly_white",
    width=800,
    height=800,
)

fig.show()
