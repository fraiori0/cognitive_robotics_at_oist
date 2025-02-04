import os
import jax
import jax.numpy as np
from jax import grad, jit, vmap
from backprop import *
from time import time
from functools import partial
import plotly.graph_objects as go

os.environ["JAX_TRACEBACK_FILTERING"] = "off"

SAVE = False


"""------------------"""
""" Model """
"""------------------"""

input_size = 2
hidden_size = 8
output_size = 2

train_seq_length = 10

SEED_PARAMS = 0

rnn = RNN(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    train_seq_length=train_seq_length,
    seed=SEED_PARAMS,
)

params = rnn.params

# print(params)


"""------------------"""
""" Data """
"""------------------"""

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
seq_p, seq_theta = gen_seq(key, state, length=200)
print("Time to generate data:", time() - t0)


# plot the data on the x=y plane
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=seq_p[:, 0],
        y=seq_p[:, 1],
        mode="markers+lines",  # "markers" #
        # using a colorscale to represent time,
        # the first point is dark blue, the last is dark red
        marker=dict(
            color=np.arange(len(seq_p)),
            colorscale="Viridis",
            colorbar=dict(title="Time"),
            size=5,
        ),
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
