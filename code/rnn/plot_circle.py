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

data_name = "eight"
hidden_size = 8

SEED = 0
key = jax.random.key(SEED)
n_steps = 30

"""------------------"""
""" Model """
"""------------------"""

input_size = 2

output_size = 2

train_seq_length = 9

SEED_PARAMS = 0

rnn = RNN(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    train_seq_length=train_seq_length,
    seed=SEED_PARAMS,
)

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
        rnn.params = pickle.load(f)
    with open(os.path.join(model_path, f"{model_name}_info.json"), "r") as f:
        model_info_dict = json.load(f)
else:
    print("Could not load parameters")
    raise ValueError

for k in rnn.params.keys():
    for kk in rnn.params[k].keys():
        print(f"{k}, {kk}, {rnn.params[k][kk].shape}")

# exit()

"""------------------"""
""" Generate trajectories using the model """
"""------------------"""

theta = np.linspace(0, 2 * np.pi, 16)
r = np.linspace(0.1, 0.5, 16)
c = np.array([0.5, 0.5])

x0 = np.concatenate(
    [
        (r[:, None] * np.cos(theta)[None, :] + c[0])[..., None],
        (r[:, None] * np.sin(theta)[None, :] + c[1])[..., None],
    ],
    axis=-1,
).reshape(-1, 2)


h0 = rnn.gen_hidden_state(key, x0.shape[0])

forward_jit = jit(rnn.forward)


# iterate  for n_steps
x = x0
h = h0

x_traj = [x]

for i in range(n_steps):
    x, h = forward_jit(rnn.params, x, h)
    x_traj.append(x)


x_traj = np.stack(x_traj, axis=1)


"""------------------"""
""" Plot """
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
    # amnd axis ranges in 0-1
    xaxis_range=[0, 1],
    yaxis_range=[0, 1],
)

fig.show()
