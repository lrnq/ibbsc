import numpy as np
import torch
from scipy import stats


def truncated_normal(tensor):
    # TODO: Fix uglyness 
    cuda = tensor.is_cuda
    if cuda:
        tensor = tensor.cpu().detach().numpy()
    else:
        tensor = tensor.detach().numpy()
    a, b = -2, 2
    X = stats.truncnorm(a,b, loc=0, scale=1/float(np.sqrt(tensor.shape[0])))
    truncnorm = X.rvs(tensor.shape)
    if cuda:
        return torch.from_numpy(truncnorm).cuda()
    else:
        return torch.from_numpy(truncnorm)


def truncated_normal_(tensor):
    """
    Helper function to generate a truncated normal distribution 
    for initializing layers as done in (Ziv, Tishby. 2017) 
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