"""
This is the main function that can generate and save the relevant data to be plotted 
with the functions in generate_plots_thesis.py. 

An example of how to run it is shown at the bottom.
"""

import sys
sys.path.insert(0,'../ibbsc')

from trainer import Trainer
from mutual_inf import MI
import argparse
import data_utils
from torch import optim
import info_utils
import numpy as np
from models import FNN
import torch 
from torch import nn
import pickle
import tqdm
import os 
import default_params


def check_for_data(save_path):
    if os.path.isdir(save_path):
        resp = input("The folder to save data in already exist. Type \"yes\" to continue: ")
        if resp == "yes":
            pass
        else:
            raise  Exception("Data already exists... Quitting...")
    else:
        os.mkdir(save_path)


def prepare_data(data_path, test_size, seed, batch_size):
    X_train, X_test, y_train, y_test = data_utils.load_data(data_path, test_size, seed)

    # Prepare data for pytorch
    if batch_size != "full":
        train_loader = data_utils.create_dataloader(X_train, y_train, batch_size, seed)
        test_loader = data_utils.create_dataloader(X_test, y_test, batch_size, seed)
    else:
        train_loader = data_utils.create_dataloader(X_train, y_train, len(X_train), seed)
        test_loader = data_utils.create_dataloader(X_test, y_test, len(X_test), seed)

    # Activitiy loaders
    full_X, full_y = np.concatenate((X_train, X_test)), np.concatenate((y_train, y_test))
    act_full_loader = data_utils.create_dataloader(full_X, full_y, len(full_X), seed, shuffle=False)

    return train_loader, test_loader, act_full_loader



def main_func(activation, data_path, save_path, batch_size, epochs, layer_sizes, mi_methods, num_bins=[30], num_runs=1, try_gpu=False):
    
    check_for_data(save_path)

    if try_gpu:
        cuda = torch.cuda.is_available() 
        device = torch.device("cuda" if cuda else "cpu")
    else:
        device = torch.device("cpu")
    print("Using "+ str(device))

    loss_function = nn.CrossEntropyLoss() # Only one supported as of now
    #max_values = []


    for i in tqdm.tqdm(range(num_runs)):
        torch.manual_seed(i)
        np.random.seed(i)
        
        train_loader, test_loader, act_full_loader = prepare_data(data_path, 819, i, batch_size)

        model = FNN(layer_sizes, activation=activation, seed=i).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0004)
        tr = Trainer(loss_function, epochs, model, optimizer, device)
        tr.train(train_loader, test_loader, act_full_loader)
        with open(save_path + '/training_history_run_{}_{}.pickle'.format(i, batch_size), 'wb') as f:
            pickle.dump([tr.error_train, tr.error_test], f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()

        #with open(save_path + '/max_values{}_{}.pickle'.format(i, batch_size), 'wb') as f:
        #    pickle.dump(tr.max_value_layers, f, protocol=pickle.HIGHEST_PROTOCOL)
        #    f.close()

        for j in num_bins:
            if "variable" in mi_methods:
                max_value = info_utils.get_max_value(tr.hidden_activations)
                num_bins = int(max_value*15)
                mutual_inf = MI(tr.hidden_activations, act_full_loader,act=activation, num_of_bins=j)
                MI_XH, MI_YH = mutual_inf.get_mi(method="fixed")
                with open(save_path + '/MI_XH_MI_YH_run_{}_{}_{}variable.pickle'.format(i, batch_size, j), 'wb') as f:
                    pickle.dump([MI_XH, MI_YH], f, protocol=pickle.HIGHEST_PROTOCOL)
                    f.close()


            if "fixed" in mi_methods:
                mutual_inf = MI(tr.hidden_activations, act_full_loader,act=activation, num_of_bins=j)
                MI_XH, MI_YH = mutual_inf.get_mi(method="fixed")

                with open(save_path + '/MI_XH_MI_YH_run_{}_{}_{}bins.pickle'.format(i, batch_size, j), 'wb') as f:
                    pickle.dump([MI_XH, MI_YH], f, protocol=pickle.HIGHEST_PROTOCOL)
                    f.close()
            
            if "adaptive" in mi_methods:
                mutual_inf = MI(tr.hidden_activations, act_full_loader,act=activation, num_of_bins=j)
                MI_XH, MI_YH = mutual_inf.get_mi(method="adaptive")

                with open(save_path + '/MI_XH_MI_YH_run_{}_{}_{}adaptive.pickle'.format(i, batch_size, j), 'wb') as f:
                    pickle.dump([MI_XH, MI_YH], f, protocol=pickle.HIGHEST_PROTOCOL)
                    f.close()

        #max_values.append(mutual_inf.max_val)
        #print(max_values)

        # Need to delete everything from memory
        # because python will keep things in memory until computation of overwriting
        # variable is finished for the next iteration. This simply fills up my RAM.
        del model
        del tr
        del mutual_inf
        del train_loader
        del test_loader
        del MI_XH
        del MI_YH
        del act_full_loader
    print("Done runnning...")


if __name__ == "__main__":
    args = default_params.default_params()
    print(args)
    main_func(args.activation, args.data, args.save_path, args.batch_size, args.epochs,
             args.layer_sizes, args.mi_methods, args.num_bins, args.num_runs, args.try_gpu)