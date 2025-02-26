# cognitive_robotics_at_oist

In the scripts training the RNN and plotting (which actually deploy the RNN for inference inside the script), there are two versions.
2.0: consider a single common initial hidden state $h(-1)$, at most separated between different trajectories (e.g. in the two circle case, we can have two separate initial hidden states)
2.1: consider a separate initial hidden state for each training mini-sequence

Both variants can backpropagate on the (possibly shared) value of the initial hidden state or use a random initial hidden state.

The plot script should correspond to the training version (i.e., plot_2.0.*.py for networks trained with train_2.0.py), because the dictionary of initial hidden states is saved with different keys in the two cases.

Note: training is always done on mini-sequences generated from a moving window applied to each of the single original trajectories.