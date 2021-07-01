import numpy as np


def get_min_max_vals(act, activity):
    """
    Determine the max and min value for the binning range when the uniform  binning
    strategy is used.

    Args:
        act: activation function
        activity: saved activation values

    Returns:
        min_val, max_val: minimum and maximum value for the binning range based on max actication value.
    """
    if act == "tanh":
        min_val = -1
        max_val = 1
    elif act == "elu":
        min_val = -1
        max_val = get_max_value(activity)
    elif act == "relu6":
        min_val = 0
        max_val = 6
    else:
        min_val = 0
        max_val = get_max_value(activity)
    return min_val, max_val


def get_max_value(activity):
    """
    Finds the maximum value over all the saved activation values
    Args:
        activity: activation values

    Returns:
        max_val: the largest activation value
    """
    max_val = 0
    for i in activity:
        for j in i:
            mm = j.max()
            if mm > max_val:
                max_val = mm
    return max_val


def get_max_val_all_layers(activations):
    """
    Finds the maximum activation value for all layers over all epochs 
    of the saved activation values.

    Args:
        activations: saved activation values

    Returns:
        max_vals: maximum activation value for all layers over all epochs
    """
    max_vals = [] # shape of (num_epochs, num_layers)
    for epoch in activations:
        max_epoch = []
        for layer in epoch: 
            max_epoch.append(layer.max()) 
        max_vals.append(max_epoch)
    return max_vals


def get_bins_layers(activations, num_bins, act):
    """
    Finds the bin borders for each layer for each epoch  when the adaptive
    binning strategy is used.It ensures that each bin contains equally many
    unique activation values.

    Args:
        activations: saved activation values
        num_bins: number of bins to produce
        act: the activation function used

    Returns:
        bins: array with bin borders for each layer for each epoch
    """
    bins = []
    for epoch in activations:
        epoch_bins=[]
        for layer in epoch:
            if act == "linear":
                lb_val = layer.min()
                lb = [lb_val] # min value possible 
            elif act in ["tanh", "elu"]:
                lb=[-1.000000000001] # min value possible
            else:
                lb=[0] # min value possible 
            unique_act_vals=np.unique(layer.flatten())  # layer.flatten() is of shape (num_samples, size_layer)
            sorted_ua = np.sort(np.setdiff1d(unique_act_vals,lb))
            if len(sorted_ua)>0: 
                last_idx = np.floor((((num_bins-1)*(len(sorted_ua))) / num_bins))
                inds = list(map(int, np.linspace(0, last_idx, num_bins)))
                borders = list(map(lambda x: sorted_ua[x], inds))
                lb.extend(borders)
                lb.append(sorted_ua[-1])
            epoch_bins.append(np.array(lb))
        bins.append(epoch_bins)    
    return np.array(bins)
