import numpy as np


def get_min_max_vals(activation_func, activity):
    if activation_func == "tanh":
        min_val = -1
        max_val = 1
    elif activation_func == "elu":
        min_val = -1
        max_val = get_max_value(activity)
    elif activation_func == "relu6":
        min_val = 0
        max_val = 6
    else:
        min_val = 0
        max_val = get_max_value(activity)
    return min_val, max_val



def get_max_value(activity):
    max_val = 0
    for i in activity:
        for j in i:
            mm = j.max()
            if mm > max_val:
                max_val = mm
    return max_val



def get_max_val_all_layers(activations):
    max_vals = [] # shape of (num_epochs, num_layers)
    for epoch in activations:
        max_epoch = []
        # after 1 epoch the whole dataset is sent through the model
        # here we loop over each layer for the pass through after an epoch.
        for layer in epoch: 
            # append max value for that layer. to max_epoch that is of length num_layers
            max_epoch.append(layer.max()) 
        max_vals.append(max_epoch)
    return max_vals



def get_bins_layers(activations, num_bins, act):
    bins = []
    for epoch in activations:
        epoch_bins=[]
        for layer in epoch:
            if act == "linear":
                lb_val = layer.min()
                lb = [lb_val]
            elif act in ["tanh", "elu"]:
                lb=[-1.000000000001] # min value possible
            else:
                lb=[0] # min value possible 
            # layer.flatten() is of shape (num_samples, size_layer)
            # Find all unique values in the layer over all samples
            unique_act_vals=np.unique(layer.flatten()) 
            # Get values that is not the min value
            sorted_ua = np.sort(np.setdiff1d(unique_act_vals,lb))
            #sorted_ua = np.array(sorted(list(set(unique_act_vals) - set(lb))))
            if len(sorted_ua)>0: 
                last_idx = np.floor((((num_bins-1)*(len(sorted_ua))) / num_bins))
                inds = list(map(int, np.linspace(0, last_idx, num_bins)))
                borders = list(map(lambda x: sorted_ua[x], inds))
                lb.extend(borders)
                lb.append(sorted_ua[-1])
            epoch_bins.append(np.array(lb))
        bins.append(epoch_bins)    
    return np.array(bins)
