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

data_name = "two_circles_opp"
hidden_size = 16

SEED = 3
key = jax.random.key(SEED)
n_steps = 200

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

name_suffix = f"_{data_name}_{hidden_size}"
model_name = "rnn" + name_suffix

# import the model's parameters if they exist
if os.path.exists(os.path.join(model_path, f"{model_name}_params.pkl")):
    with open(os.path.join(model_path, f"{model_name}_params.pkl"), "rb") as f:
        params = pickle.load(f)
    with open(os.path.join(model_path, f"{model_name}_info.json"), "r") as f:
        model_info_dict = json.load(f)
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
    hidden_activation_fn_grad=activation_dict[model_info_dict["hidden_activation"]]["grad"],
    output_activation_fn=activation_dict[model_info_dict["output_activation"]]["fn"],
    output_activation_fn_grad=activation_dict[model_info_dict["output_activation"]]["grad"],
    seed=SEED_PARAMS,
)

# Set parameters
rnn.params = params


for k in rnn.params.keys():
    for kk in rnn.params[k].keys():
        print(f"{k}, {kk}, {rnn.params[k][kk].shape}")


# exit()

"""------------------"""
""" Generate trajectories using the model """
"""------------------"""

# sample points homogenously in x-y, range 0-1
x0 = np.array(
    np.meshgrid(
        np.linspace(0, 1, 11),
        np.linspace(0, 1, 11),
    )
).T.reshape(-1, 2)

# generate initial hidden states
h0 = rnn.gen_hidden_state_uniform(key, x0.shape[0], h0_min, h0_max)

forward_jit = jit(rnn.forward)


# iterate  for n_steps
x = x0
h = h0

x_traj = [x]
h_traj = [h]

for i in range(n_steps):
    x, h = forward_jit(rnn.params, x, h)
    x_traj.append(x)
    h_traj.append(h)


x_traj = np.stack(x_traj, axis=1)
h_traj = np.stack(h_traj, axis=1)

print(x_traj.shape)
print(h_traj.shape)

# exit()


"""------------------"""
""" PCA """
"""------------------"""

# perform PCA on the hidden states
pca = PCA(n_components=2)
# fit to (flattened) hidden states
pca.fit(onp.array(h_traj.reshape(-1, h_traj.shape[-1])))


# take the components and convert to JAX array
# note, it's dimensions are (n_components, n_features)
pca_components = np.array(pca.components_)

# project the (original) hidden states onto the PCA components
# so we keep the dimensions as (n_traj, n_steps, features)

h_traj_pca = (h_traj[..., None, :] * pca_components).sum(axis=-1)


"""------------------"""
""" Plot Trajectories """
"""------------------"""

# plot all the trajectories

fig = go.Figure()

for i in range(x0.shape[0]):
    fig.add_trace(
        go.Scatter(
            x=x_traj[i, :, 0],
            y=x_traj[i, :, 1],
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

# exit()


"""------------------"""
""" Plot PCA """
"""------------------"""

# plot the trajectories of the hidden state as projected on the PCA components
fig = go.Figure()

for i in range(h_traj.shape[0]):
    fig.add_trace(
        go.Scatter(
            x=h_traj_pca[i, :, 0],
            y=h_traj_pca[i, :, 1],
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
