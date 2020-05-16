import sys
sys.path.insert(0,'..')

import torch
from torch.utils import data
from torchvision import datasets
import scipy.io as sio
from trainer import Trainer
from models import FNN
from mutual_inf import MI
import plot_utils
import data_utils
from random import seed
import numpy as np
import utils
import torch 
from torch import nn
import pickle
import tqdm



# Run on GPU if possible
try_gpu = True
if try_gpu:
    cuda = torch.cuda.is_available() 
    device = torch.device("cuda" if cuda else "cpu")
else:
    device = torch.device("cpu")
print("Using "+ str(device))
activation = "tanh"

train_data = datasets.MNIST("../data", train=True, download=True) 
test_data = datasets.MNIST("../data", train=False, download=True) 

X_train, y_train = (train_data.data.numpy()/255).reshape(-1, 28*28), train_data.targets.numpy()
X_test, y_test = (test_data.data.numpy()/255).reshape(-1, 28*28), test_data.targets.numpy()

# Useless to have a config contain these arbitrary things. 
config = {
"loss_function" : nn.CrossEntropyLoss(),
"batch_size" : 128,
"epochs" : 10000,
"layer_sizes" : [784, 1024, 20, 20, 20, 10]
}
layer_sizes = [784, 1024, 20, 20, 20, 10]

ext = ""
ext += str(config["batch_size"]) + str(config["epochs"]) + str(config["layer_sizes"]) + "MNIST" + activation
print(ext)

for i in tqdm.tqdm(range(1)):
    np.random.seed(i)
    torch.manual_seed(i)

    # Prepare data for pytorch
    if config["batch_size"] != "full":
        train_loader = data_utils.create_dataloader(X_train, y_train, config["batch_size"])
        test_loader = data_utils.create_dataloader(X_test, y_test, config["batch_size"])
    else:
        train_loader = data_utils.create_dataloader(X_train, y_train, len(X_train))
        test_loader = data_utils.create_dataloader(X_test, y_test, len(X_test))


    # Activitiy loaders
    full_X, full_y = np.concatenate((X_train, X_test)), np.concatenate((y_train, y_test))
    act_full_loader = data_utils.create_dataloader(full_X, full_y, len(full_X))
    act_train_loader = data_utils.create_dataloader(X_train, y_train, len(X_train))
    #act_test_loader = data_utils.create_dataloader(X_test, y_test, len(X_test))
    act_loaders = [act_full_loader]

    ib_model = FNN(layer_sizes, seed=i,  activation=activation).to(device)
    optimizer = torch.optim.Adam(ib_model.parameters(), lr=0.001)
    tr = Trainer(config, ib_model, optimizer, device)
    tr.train(train_loader, test_loader, act_loaders)
    mutual_inf = MI(tr.hidden_activations, act_full_loader,act=activation, num_of_bins=30)
    MI_XH, MI_YH = mutual_inf.get_MI()
    with open('../data/run_{}_{}.pickle'.format(i, ext), 'wb') as f:
        pickle.dump([MI_XH, MI_YH], f, protocol=pickle.HIGHEST_PROTOCOL)