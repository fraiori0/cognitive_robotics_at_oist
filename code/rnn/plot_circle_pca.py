import os
import jax
import jax.numpy as np
from jax import grad, jit, vmap
import numpy as onp
from backprop import *
from time import time
from functools import partial
import plotly.graph_objects as go
import json
import pickle
from sklearn.decomposition import PCA

os.environ["JAX_TRACEBACK_FILTERING"] = "off"

SAVE = True

data_name = "circle"
test_name = "h0"
hidden_size = 2

SEED = 3
key = jax.random.key(SEED)
n_steps = 50

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
else:
    print("Could not load parameters")
    raise ValueError("Could not load parameters")

h0_min = model_info_dict["h0_min"]
h0_max = model_info_dict["h0_max"]

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

SEED_PARAMS = 0

rnn = RNN(
    input_size=model_info_dict["input_size"],
    hidden_size=hidden_size,
    output_size=model_info_dict["output_size"],
    train_seq_length=model_info_dict["train_seq_length"],
    hidden_activation_fn=activation_dict[model_info_dict["hidden_activation"]]["fn"],
    hidden_activation_fn_grad=activation_dict[model_info_dict["hidden_activation"]][
        "grad"
    ],
    output_activation_fn=activation_dict[model_info_dict["output_activation"]]["fn"],
    output_activation_fn_grad=activation_dict[model_info_dict["output_activation"]][
        "grad"
    ],
    seed=SEED_PARAMS,
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
    print(f"\t\tX0: {hidden_states[k]['x0'].shape}")
    print(f"\t\tH0: {hidden_states[k]['h0'].shape}")


# exit()

"""------------------"""
""" Generate trajectories using the model """
"""------------------"""

# sample points homogenously in x-y, range 0-1
x0 = np.array(
    np.meshgrid(
        np.linspace(0, 1, 6),
        np.linspace(0, 1, 6),
    )
).T.reshape(-1, 2)

# for each key (i.e., different training sequence)
# take the initial hidden state as the one associated to the point (x) closest to
# starting point x0
h0_dict = {}

for k in hidden_states.keys():
    # compute distances
    ds = x0[:, None] - hidden_states[k]["x0"][None, :]
    ds = ds**2
    ds = ds.sum(axis=-1)
    # convert them to the index of the point in x0_saved with the minimum distance
    ds = ds.argmin(axis=-1)
    # take the initial hidden state associated to such point
    h0_dict[k] = hidden_states[k]["h0"][ds]
    print(h0_dict[k])


forward_jit = jit(rnn.forward)


# for each key (i.e., different training sequence)
# start from the given conditions (x0 and h0)
# and make n_steps (note, we batch over the different starting points)

x_traj = {}
h_traj = {}

for k in h0_dict.keys():

    x = x0
    h = h0_dict[k]

    x_traj[k] = [x]
    h_traj[k] = [h]

    for i in range(n_steps):
        x, h = forward_jit(rnn.params, x, h)
        x_traj[k].append(x)
        h_traj[k].append(h)

# convert to numpy array
for k in x_traj.keys():
    x_traj[k] = np.stack(x_traj[k], axis=1)
    h_traj[k] = np.stack(h_traj[k], axis=1)

# print shape
for k in x_traj.keys():
    print(f"\n{k}")
    print(f"\tx_traj[{k}]: {x_traj[k].shape}")
    print(f"\th_traj[{k}]: {h_traj[k].shape}")


"""------------------"""
""" PCA """
"""------------------"""

# perform PCA on the hidden states
pca = PCA(n_components=2)

pca_components_dict = {}

for k in h_traj.keys():
    # fit to (flattened) hidden states
    pca.fit(onp.array(h_traj[k].reshape(-1, h_traj[k].shape[-1])))
    # take the components and convert to JAX array
    # note, it's dimensions are (n_components, n_features)
    pca_components = np.array(pca.components_)
    # store the components
    pca_components_dict[k] = pca_components


# project the (original) hidden states onto the PCA components
# so we keep the dimensions as (n_traj, n_steps, features)

h_traj_pca = {}
for k in h_traj.keys():
    h_traj_pca[k] = (h_traj[k][..., None, :] * pca_components_dict[k]).sum(axis=-1)

# print shape
for k in h_traj_pca.keys():
    print(f"\n{k}")
    print(f"\th_traj_pca[{k}]: {h_traj_pca[k].shape}")

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
for i, k in enumerate(x_traj.keys()):
    color_dict[k] = my_colors[i]


# plot all the trajectories
# in state space (x-y)

fig = go.Figure()

for k in x_traj.keys():
    for i in range(x0.shape[0]):
        fig.add_trace(
            go.Scatter(
                x=x_traj[k][i, :, 0],
                y=x_traj[k][i, :, 1],
                mode="markers+lines",  # $"markers+lines",  #
                # using a colorscale to represent time,
                # the first point is dark blue, the last is dark red
                marker=dict(
                    color=np.arange(n_steps + 1),
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
    xaxis_range=[0, 1],
    yaxis_range=[0, 1],
    # set white background and style
    # plot_bgcolor="white",
    template="plotly_white",
)


# add ticks at 0.1 marks, both on x-y axis, and only in the range 0-1
fig.update_xaxes(tick0=0, dtick=0.1, range=[0, 1])
fig.update_yaxes(tick0=0, dtick=0.1, range=[0, 1])
# set tick color to black


fig.show()


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
                    color=np.arange(n_steps + 1),
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
)

fig.show()

exit()

"""------------------"""
""" PCA with 3 components """
"""------------------"""

# if there are more than 2 hidden features, perform PCA with 3 components

if h_traj.shape[-1] < 3:
    exit()

# perform PCA on the hidden states
pca = PCA(n_components=3)
# fit to (flattened) hidden states
pca.fit(onp.array(h_traj.reshape(-1, h_traj.shape[-1])))
# take the components and convert to JAX array
# note, it's dimensions are (n_components, n_features)
pca_components = np.array(pca.components_)
# project the (original) hidden states onto the PCA components
# so we keep the dimensions as (n_traj, n_steps, features)
h_traj_pca = (h_traj[..., None, :] * pca_components).sum(axis=-1)

# plot the trajectories of the hidden state as projected on the PCA components
# make a 3D plot

fig = go.Figure()

for i in range(h_traj.shape[0]):
    fig.add_trace(
        go.Scatter3d(
            x=h_traj_pca[i, :, 0],
            y=h_traj_pca[i, :, 1],
            z=h_traj_pca[i, :, 2],
            mode="markers+lines",  # $"markers+lines",  #
            # using a colorscale to represent time,
            # the first point is dark blue, the last is dark red
            marker=dict(
                color=np.arange(n_steps + 1),
                colorscale="Viridis",
                colorbar=dict(title="Time"),
                size=5,
            ),
            # line of same color and transparency
            line=dict(color="rgba(135, 206, 250, 0.4)"),
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
)

fig.show()
