import numpy as np
import torch


def truncated_normal_(tensor):
    """
    Helper function to generate a truncated normal distribution 
    for initializing layers.
    source: https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/2
    """
    mean = 0
    std = 1/float(np.sqrt(tensor.shape[0]))
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor