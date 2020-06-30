import sys
from trainer import Trainer
from mutual_inf import MI
import argparse
import data_utils
from torch import optim
import info_utils
import numpy as np
from models import FNN
import plot_utils
import torch 
from torch import nn
import pickle
import tqdm
import os 
import default_params


def check_for_data(save_path):
    """
    Check if the data exists already, and prompt the user to confirm the path
    to prevent overwriting already saved data.
    Args:
        save_path: path to save the mutual information and plots.

    Returns:
        0
    """
    if os.path.isdir(save_path):
        resp = input("The folder to save data in already exist. Type \"yes\" to continue: ")
        if resp == "yes":
            pass
        else:
            raise  Exception("Data already exists... Quitting...")
    else:
        os.mkdir(save_path)


def prepare_data(data_path, test_size, seed, batch_size):
    """
    Prepare the dataloaders for the training. These are also passed to the MI
    constructor and unpacked in the MI class to compute the mutual information
    Args:
        data_path: path to load the dataset from.
        test_size: percentage of data to use as test (or number of samples).
        seed: rng seed.
        batch_size: batch size to use.

    Returns:
        [type]: [description]
    """
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



def main_func(activation, data_path, save_path, batch_size, epochs, layer_sizes, mi_methods, test_size, num_bins=[30], num_runs=1, try_gpu=False):
    
    check_for_data(save_path)

    if try_gpu:
        cuda = torch.cuda.is_available() 
        device = torch.device("cuda" if cuda else "cpu")
    else:
        device = torch.device("cpu")
    print("Using "+ str(device))

    loss_function = nn.CrossEntropyLoss() # Only one supported as of now
    max_values = []


    for i in tqdm.tqdm(range(args.start_from, num_runs)):
        torch.manual_seed(i)
        torch.cuda.manual_seed(i)
        np.random.seed(i)
        
        train_loader, test_loader, act_full_loader = prepare_data(data_path, test_size, i, batch_size)

        model = FNN(layer_sizes, activation=activation, seed=i).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0004)
        tr = Trainer(loss_function, epochs, model, optimizer, device)
        print("Start Training...")
        tr.train(train_loader, test_loader, act_full_loader)

        if args.save_train_error:
            print("Saving train and test error...")
            with open(save_path + '/training_history_run_{}_{}.pickle'.format(i, batch_size), 'wb') as f:
                pickle.dump([tr.error_train, tr.error_test], f, protocol=pickle.HIGHEST_PROTOCOL)
                f.close()
            with open(save_path + '/loss_run_{}_{}.pickle'.format(i, batch_size), 'wb') as f:
                pickle.dump([tr.train_loss, tr.val_loss], f, protocol=pickle.HIGHEST_PROTOCOL)
                f.close()

        if args.save_max_vals:
            print("Saving max activation values...")
            with open(save_path + '/max_values{}_{}.pickle'.format(i, batch_size), 'wb') as f:
                print(np.array(tr.max_value_layers_mi).max())
                pickle.dump(tr.max_value_layers_mi, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.close()

        if args.save_mutual_information:
            for j in num_bins:
                print("Saving mutual information with {} bins...".format(j))
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

        minv, maxv = info_utils.get_min_max_vals(activation, tr.hidden_activations)
        max_values.append(maxv)
        print(max_values)

        # Need to delete everything from memory
        # because python will keep things in memory until computation of overwriting
        # variable is finished for the next iteration. This simply fills up my RAM.
        del model
        del tr
        if args.save_mutual_information:
            del mutual_inf
            del MI_XH
            del MI_YH
        del train_loader
        del test_loader
        del act_full_loader
    print("Done runnning...")


if __name__ == "__main__":
    args = default_params.default_params()
    print(args)
    print("Running main function...")
    main_func(args.activation, args.data, args.save_path,
              args.batch_size, args.epochs, args.layer_sizes,
              args.mi_methods, args.test_size, args.num_bins,
              args.num_runs, args.try_gpu)
    if args.plot_results:
        print("Begin plotting...")
        if not os.path.isdir(args.save_path):
            os.mkdir(args.save_path)
        ext = str(args.batch_size) + "_"
        exts = []
        for method in args.mi_methods:
            for bins in args.num_bins:
                if method == "fixed":
                    exts.append(ext + str(bins) + "bins")
                else:
                    exts.append(ext + str(bins) + str(method))

        if args.save_path[-1] != "/":
            args.save_path += "/"
        for ext in exts:
            # Read in all MI data from different runs
            full_MI_XH = np.zeros(args.num_runs,  dtype=object)
            full_MI_YH = np.zeros(args.num_runs,  dtype=object)
            for i in range(args.num_runs):
                with open(args.save_path + 'MI_XH_MI_YH_run_{}_{}.pickle'.format(i,ext), 'rb') as f:
                    MI_XH, MI_YH = pickle.load(f)
                    full_MI_XH[i] = np.array(MI_XH)
                    full_MI_YH[i] = np.array(MI_YH)

            avg_MI_XH = np.mean(full_MI_XH, axis = 0)
            avg_MI_YH = np.mean(full_MI_YH, axis = 0)

            plot_utils.plot_layer_MI(avg_MI_XH[:], "$I(X;T)$", save_path=args.save_path + "MI_XT" + ext + str(args.activation) + ".png")
            plot_utils.plot_layer_MI(avg_MI_YH[:], "$I(Y;T)$", save_path=args.save_path + "MI_YT" + ext + str(args.activation) + ".png")
            plot_utils.plot_info_plan(avg_MI_XH[:], avg_MI_YH[:], cbar_epochs=str(args.epochs), save_path=args.save_path + "infoplane" + ext + str(args.activation) + ".png")
