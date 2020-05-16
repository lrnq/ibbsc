import numpy as np


def get_max_value(activity):
    """
    Activity is an array for the activity 
    for each layer 
    """
    max_val = 0
    for i in activity:
        for j in i:
            mm = j.max()
            if mm > max_val:
                max_val = mm
    return max_val



def get_max_val_all_layers(activations):
    max_vals = []
    for epoch in activations:
        max_epoch = []
        for layer in epoch:
            max_epoch.append(layer.max())
        max_vals.append(max_epoch)
    return max_vals



def get_bins_layers(activations, num_bins, act):
    max_vals = get_max_val_all_layers(activations)
    bins = []
    if act == "tanh" or act == "elu":
        low = -1
    else:
        low = 0
    for epoch in activations:
        epoch_bins=[]
        for layer in epoch:
            layer_bins=[low]
            unique_act_vals=np.unique(layer.flatten())
            sorted_ua = np.sort(np.setdiff1d(unique_act_vals,layer_bins)) # sorted unique activations not in layer_bins
            if sorted_ua.size>0:
                for k in range(num_bins):
                    ind=int(k*(sorted_ua.size/num_bins))
                    layer_bins.append(sorted_ua[ind])
                layer_bins.append(sorted_ua[-1])
            epoch_bins.append(np.asarray(layer_bins))
        bins.append(epoch_bins)    
    return max_vals, np.array(bins)
