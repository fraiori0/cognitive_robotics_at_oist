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

SAVE = False

data_name = "two_circles_opp"

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


# print the length of the sequences for every file
print(f"\tData processed, the following training sequences are available:")
for k in ps_dict.keys():
    print(f"\t\t{k}: {ps_dict[k].shape}")


"""------------------"""
""" Visualize Trajectories """
"""------------------"""

# plot each trajectory in ps_dict

# create a figure
fig = go.Figure()

# add each trajectory to the figure
for k in ps_dict.keys():
    fig.add_trace(
        go.Scatter(
            x=ps_dict[k][:, 0],
            y=ps_dict[k][:, 1],
            mode="markers+lines",
            marker=dict(
                size=10,
                # viridis color scale
                color=np.linspace(0, 1, ps_dict[k].shape[0]),
                colorscale="Viridis",
            ),
            line=dict(
                width=2,
                color="rgba(180,0,0,0.2)",
            ),
            name=k,
        )
    )

# set the layout
fig.update_layout(
    title=f"Trajectories",
    xaxis_title="X",
    yaxis_title="Y",
    template="plotly_white",
    xaxis=dict(scaleanchor="y", scaleratio=1),
    width=800,
    height=800,
    showlegend=True,
)

# show the figure
fig.show()
