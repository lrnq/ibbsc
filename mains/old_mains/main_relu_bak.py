import sys
sys.path.insert(0,'..')


from trainer import Trainer
from ibnet import IBNet
from mutual_inf import MI
import plot_utils
import data_utils
from random import seed
from torch import optim
import network_utils
import numpy as np
import utils
from models import FNN
import torch 
from torch import nn
import pickle
import tqdm
import os 


ext = ""
activation = "relu"
#activation = "tanh"

# dataset path
data_path = "../data/var_u.mat" # Orig IB data
save_path = "../data_relu_bak"

if not os.path.isdir(save_path):
    os.mkdir(save_path)


# Run on GPU if possible
try_gpu = True
if try_gpu:
    cuda = torch.cuda.is_available() 
    device = torch.device("cuda" if cuda else "cpu")
else:
    device = torch.device("cpu")
print("Using "+ str(device))

random_split = True


# Useless to have a config contain these arbitrary things. 
config = {
"loss_function" : nn.CrossEntropyLoss(),
"batch_size" : 256,
"epochs" : 8000,
}

layer_sizes = [12,10,7,5,4,3,2]
max_values = []

ext += str(config["batch_size"]) + str(config["epochs"]) + str(layer_sizes) + activation

for i in tqdm.tqdm(range(50)): 
    torch.manual_seed(i)
    np.random.seed(i)
    if random_split:
        X_train, X_test, y_train, y_test = data_utils.load_data(data_path)
    else:
        trn, tst = utils.get_ib_data()
        X_train, y_train, c_train = utils.tensor_casting(trn)
        X_test, y_test, c_test = utils.tensor_casting(tst)

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
    #act_train_loader = data_utils.create_dataloader(X_train, y_train, len(X_train))
    #act_test_loader = data_utils.create_dataloader(X_test, y_test, len(X_test))
    act_loaders = [act_full_loader]

    ib_model = FNN(layer_sizes, activation=activation, seed=i).to(device)
    optimizer = optim.Adam(ib_model.parameters(), lr=0.0004)
    tr = Trainer(config, ib_model, optimizer, device)
    tr.train(train_loader, test_loader, act_loaders)
    with open(save_path + '/trdata_run_{}_{}.pickle'.format(i, ext), 'wb') as f:
        pickle.dump([tr.error_train, tr.error_test], f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    max_value = network_utils.get_max_value(tr.hidden_activations)
    num_bins = int(max_value*15)
    mutual_inf = MI(tr.hidden_activations, act_full_loader,act=activation, num_of_bins=num_bins)
    MI_XH, MI_YH = mutual_inf.get_MI()

    mutual_inf2 = MI(tr.hidden_activations, act_full_loader,act=activation, num_of_bins=100)
    MI_XH2, MI_YH2 = mutual_inf2.get_MI()


    with open(save_path + '/MI_XH_MI_YH_run_{}_{}variable.pickle'.format(i, ext), 'wb') as f:
        pickle.dump([MI_XH, MI_YH], f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    with open(save_path + '/MI_XH_MI_YH_run_{}_{}100bins.pickle'.format(i, ext), 'wb') as f:
        pickle.dump([MI_XH2, MI_YH2], f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    max_values.append(mutual_inf.max_val)
    print(max_values)

    del ib_model
    del tr
    del mutual_inf
    del mutual_inf2
    del train_loader
    del test_loader
    del X_test
    del X_train
    del full_X
    del full_y
    del act_loaders
    del act_full_loader