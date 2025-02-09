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

data_name = "circle"
hidden_size = 2

SEED = 0
key = jax.random.key(SEED)
n_steps = 200

"""------------------"""
""" Model """
"""------------------"""

input_size = 2
output_size = 2
train_seq_length = 100

h0_min = 0.25
h0_max = 0.75

SEED_PARAMS = 0

rnn = RNN(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    train_seq_length=train_seq_length,
    hidden_activation_fn=sigmoid,
    hidden_activation_fn_grad=sigmoid_grad,
    output_activation_fn=sigmoid,
    output_activation_fn_grad=sigmoid_grad,
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
    raise ValueError("Could not load parameters")

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
