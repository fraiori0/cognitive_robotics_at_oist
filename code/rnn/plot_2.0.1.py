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

SAVE = False

data_name = "opposing_circles"
test_name = "singleh0"
hidden_size = 6

SEED = 3
key = jax.random.key(SEED)
n_steps_rollout = 50

checkpoint = None

random_hidden_states = True

image_folder = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    os.pardir,
    os.pardir,
    "figures",
    data_name,
    test_name,
)
if checkpoint is not None:
    image_folder = os.path.join(image_folder, f"checkpoint", f"{checkpoint}")

if not os.path.exists(image_folder):
    os.makedirs(image_folder)

fig_suffix = f"_{data_name}_{hidden_size}_{test_name}_randX0"

if random_hidden_states:
    fig_suffix += "_randh0"

if checkpoint is not None:
    fig_suffix += f"_chkpt{checkpoint}"

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
    test_name,
)


name_suffix = f"_{data_name}_{hidden_size}_{test_name}"

if checkpoint is not None:
    model_path = os.path.join(model_path, f"checkpoints")
    name_suffix += f"_{checkpoint}"


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
    train_seq_length=2,
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
""" Set of random initial x """
"""------------------"""

# generate a set of random initial x
# by sampling a grid of (x,y) positions in [0,1]x[0,1]
X0 = (
    np.array(
        np.meshgrid(
            np.linspace(0, 1, 10),
            np.linspace(0, 1, 10),
        )
    )
    .reshape(2, -1)
    .T
)

print(f"\nRandom initial X0:")
print(f"\tX0: {X0.shape}")

"""------------------"""
""" Generate initial hidden states """
"""------------------"""

# Random initial hidden states
if random_hidden_states:
    key, _ = jax.random.split(key)
    H0 = rnn.gen_hidden_state_uniform(
        key,
        X0.shape[0],
        min=model_info_dict["h0_min_activation"],
        max=model_info_dict["h0_max_activation"],
    )
else:
    # randomly take the initial hidden states from the training trajectories
    H0 = np.array([hidden_states[k]["h0"] for k in hidden_states.keys()])
    # randomly take value from H0 to create X0.shape[0] initial hidden states
    key, _ = jax.random.split(key)
    idx = jax.random.randint(key, (X0.shape[0],), 0, H0.shape[0])
    H0 = H0[idx]


"""------------------"""
""" Jit Forward and Backward """
"""------------------"""

forward_jit = jit(rnn.forward)


"""------------------"""
""" Generate sequences """
"""------------------"""

# generate sequences
# start from the first x in each sequence
# and from the initial hidden states computed for each sequence
# and iterate in feedback mode for n_steps_rollout steps

x = X0
h = H0

# list to store the generated sequences
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
x_traj = np.stack(x_traj_list, axis=1)
h_traj = np.stack(h_traj_list, axis=1)

print(f"\nGenerated sequences")
print(f"\tx_traj: {x_traj.shape}")
print(f"\th_traj: {h_traj.shape}")


"""------------------"""
""" PCA """
"""------------------"""

print(f"\nPerforming PCA")

# perform PCA on the hidden states
pca = PCA(n_components=2)

pca_components_dict = {}


# fit to (flattened) hidden states
pca.fit(onp.array(h_traj.reshape(-1, h_traj.shape[-1])))
# take the components and convert to JAX array
# note, it's dimensions are (n_components, n_features)
pca_components = np.array(pca.components_)


# project the hidden states onto the PCA components
# so we keep the dimensions as (n_traj, n_steps, features)
h_traj_pca = (h_traj[..., None, :] * pca_components).sum(axis=-1)


# project also the initial hidden states
h0_pca = (H0[..., None, :] * pca_components).sum(axis=-1)


# print shape
print(f"\t{k}")
print(f"\t  h_traj_pca: {h_traj_pca.shape}")
print(f"\t  h0_pca: {h0_pca.shape}")


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


# plot all the trajectories
# in state space (x-y)

fig = go.Figure()

for i in range(x_traj.shape[0]):
    fig.add_trace(
        go.Scatter(
            x=x_traj[i, :, 0],
            y=x_traj[i, :, 1],
            mode="markers+lines",  # $"markers+lines",  #
            # using a colorscale to represent time,
            # the first point is dark blue, the last is dark red
            marker=dict(
                color=np.linspace(0, 1, x_traj.shape[1]),
                colorscale="Viridis",
                colorbar=dict(title="Time"),
                size=8,
            ),
            # set line color and size
            line=dict(
                color=my_colors[i % len(my_colors)],
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

if SAVE:
    fig.write_html(
        os.path.join(image_folder, f"x_traj_{fig_suffix}.html"),
        include_plotlyjs="directory",
    )

# exit()

"""------------------"""
""" Plot Loss History """
"""------------------"""

# plot the loss history
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        y=history["loss"],
        mode="lines",
        name="MSE loss",
        line=dict(
            color="rgba(0, 157, 255, 1)",
            width=3,
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

if SAVE:
    fig.write_html(
        os.path.join(image_folder, f"loss_history_{fig_suffix}.html"),
        include_plotlyjs="directory",
    )

# exit()

"""------------------"""
""" Plot PCA """
"""------------------"""

# plot the trajectories of the hidden state as projected on the PCA components
fig = go.Figure()

for i in range(h_traj_pca.shape[0]):
    fig.add_trace(
        go.Scatter(
            x=h_traj_pca[i, :, 0],
            y=h_traj_pca[i, :, 1],
            mode="markers+lines",  # $"markers+lines",  #
            # using a colorscale to represent time,
            # the first point is dark blue, the last is dark red
            marker=dict(
                color=np.linspace(0, 1, h_traj.shape[1]),
                colorscale="Viridis",
                colorbar=dict(title="Time"),
                size=4,
            ),
            # set line color and size
            line=dict(
                color=my_colors[i % len(my_colors)],
                width=2,
            ),
            # legendgroup=k,
        )
    )
    # add red markers for the pca of the initial hidden states
    fig.add_trace(
        go.Scatter(
            x=h0_pca[:, 0],
            y=h0_pca[:, 1],
            mode="markers",
            marker=dict(
                color="red",
                size=3,
            ),
            name="Initial hidden state",
            legendgroup="Initial hidden state",
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

if SAVE:
    fig.write_html(
        os.path.join(image_folder, f"pca2D_{fig_suffix}.html"),
        include_plotlyjs="directory",
    )

# exit()

"""------------------"""
""" PCA with 3 components """
"""------------------"""

# if there are more than 2 hidden features, perform PCA with 3 components

if h_traj.shape[-1] < 3:
    exit()
print(f"\nPerforming PCA with 3 components")

# perform PCA on the hidden states
pca = PCA(n_components=3)

# fit to (flattened) hidden states
pca.fit(onp.array(h_traj.reshape(-1, h_traj.shape[-1])))
# take the components and convert to JAX array

pca_components = np.array(pca.components_)

# project the hidden states onto the PCA components
# so we keep the dimensions as (n_traj, n_steps, features)
h_traj_pca = (h_traj[..., None, :] * pca_components).sum(axis=-1)

# project also the initial hidden states
h0_pca = (H0[..., None, :] * pca_components).sum(axis=-1)

# print shape
print(f"\t  h_traj_pca: {h_traj_pca.shape}")
print(f"\t  h0_pca: {h0_pca.shape}")


# plot the trajectories of the hidden state as projected on the PCA components
# make a 3D plot

fig = go.Figure()

for i in range(h_traj_pca.shape[0]):
    fig.add_trace(
        go.Scatter3d(
            x=h_traj_pca[i, :, 0],
            y=h_traj_pca[i, :, 1],
            z=h_traj_pca[i, :, 2],
            mode="markers+lines",  # $"markers+lines",  #
            # using a colorscale to represent time,
            # the first point is dark blue, the last is dark red
            marker=dict(
                color=np.linspace(0, 1, h_traj.shape[1]),
                colorscale="Viridis",
                colorbar=dict(title="Time"),
                size=4,
            ),
            # set line color and size
            line=dict(
                color=my_colors[i % len(my_colors)],
                width=2,
            ),
            # legendgroup=k,
        )
    )

    # add red markers for the pcs of the initial hidden states
    fig.add_trace(
        go.Scatter3d(
            x=h0_pca[:, 0],
            y=h0_pca[:, 1],
            z=h0_pca[:, 2],
            mode="markers",
            marker=dict(
                color="red",
                size=4,
            ),
            name="Initial hidden state",
            legendgroup="Initial hidden state",
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


if SAVE:
    fig.write_html(
        os.path.join(image_folder, f"pca3D_{fig_suffix}.html"),
        include_plotlyjs="directory",
    )
