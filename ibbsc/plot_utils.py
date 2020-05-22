import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import pickle
from collections import Counter
import numpy as np
import seaborn as sns
import tqdm 
sns.set()


def plot_layer_MI(MI, y_label, subplot=False, dataset="", save_plot=True, save_path=None):
    if subplot:
        subplots_num = MI.shape[1]
        fig, axs = plt.subplots(1, subplots_num, figsize=(15,5))

        for idx in range(MI.shape[1]):
            axs[idx].plot(MI[:,idx], label="Hidden Layer {}".format(idx+1))
            axs[idx].set_ylabel(y_label)
            axs[idx].set_xlabel('Epoch')
            axs[idx].legend()

        if save_plot and save_path:
            fig.savefig(save_path)
        else:
            plt.show()
        return

    fig, ax = plt.subplots(1)
    for idx in range(MI.shape[1]):
            ax.plot(MI[:,idx], label="Hidden Layer {}".format(idx+1))
            ax.set_ylabel(y_label)
            ax.set_xlabel('Epoch')
            ax.legend()
        
    plt.tight_layout()
    if save_plot and save_path:
        fig.savefig(save_path)
    else:
        plt.show()
    return 


def plot_info_plan(MI_XH, MI_YH, cbar_epochs="8000", dataset="", save_plot=True, save_path=None):
    fig, ax = plt.subplots(1)
    
    # Similar to ibsgd and idnn
    # source: source https://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib
    cmap = plt.get_cmap('gnuplot')
    norm_colors  = matplotlib.colors.Normalize(vmin=0, vmax=MI_XH.shape[0]-1)
    mapcs = matplotlib.cm.ScalarMappable(norm=norm_colors, cmap=cmap)
    colors = [mapcs.to_rgba(i) for i in range(MI_XH.shape[0])]
    ax.set_prop_cycle(color = colors)

    for j in tqdm.tqdm(range(MI_XH.shape[0])):
        ax.scatter(MI_XH[j, :], MI_YH[j, :], s=20, alpha=0.9, zorder=3)
        ax.plot(MI_XH[j, :], MI_YH[j, :], alpha=0.01, zorder=1)

    #ax.set_title('Information Plane - {}'.format(dataset))
    #ax.set_xticks(range(13)) # Hard coded for IB dataset 
    ax.set_xlabel('$I(X;T)$')
    ax.set_ylabel('$I(T;Y)$')
    cbar = fig.colorbar(mapcs, ticks=[])
    cbar.set_label("Epochs")
    #source: https://stackoverflow.com/questions/28808143/putting-tick-values-on-top-and-bottom-of-matplotlib-colorbar
    cbar.ax.text(0.5, -0.01, '0', transform=cbar.ax.transAxes, va='top', ha='center')
    cbar.ax.text(0.5, 1.0, cbar_epochs, transform=cbar.ax.transAxes, va='bottom', ha='center')
    plt.tight_layout()
    if save_plot and save_path:
        fig.savefig(save_path)
    else:
        plt.show()
    return 
    


def plot_average_MI(num_runs, ext, data_path, save_plot, save_path):
    full_MI_XH = np.zeros(num_runs,  dtype=object)
    full_MI_YH = np.zeros(num_runs,  dtype=object)
    for i in range(num_runs):
        with open(data_path + 'MI_XH_MI_YH_run_{}_{}.pickle'.format(i,ext), 'rb') as f:
            MI_XH, MI_YH = pickle.load(f)
            full_MI_XH[i] = np.array(MI_XH)
            full_MI_YH[i] = np.array(MI_YH)

    avg_MI_XH = np.mean(full_MI_XH, axis = 0)
    avg_MI_YH = np.mean(full_MI_YH, axis = 0)


    plot_info_plan(avg_MI_XH[:], avg_MI_YH[:], save_plot=save_plot, save_path=save_path)
    plot_layer_MI(avg_MI_XH[:], "$I(X;T)$",  save_plot=save_plot, save_path=save_path+"XH")
    plot_layer_MI(avg_MI_YH[:], "$I(Y;T)$",  save_plot=save_plot, save_path=save_path + "YH")

    return



def plot_max_vals(max_vals, save_path):
    counts = Counter(max_vals)
    fig, ax = plt.subplots(1)
    ax.set_ylabel("Max Activation Value")
    ax.set_xlabel("Network")

    ax.bar(range(len(counts.values())), counts.keys(), width=1)

    fig.savefig(save_path)
    return 


def plot_binning_methods(activation_function, bin_boundaries):
    """
    Currently in a notebook
    """
    raise NotImplementedError




def plot_subset_runs_info(runs, ext, save_path, data_path, save_plot=True):
    full_MI_XH = np.zeros(len(runs),  dtype=object)
    full_MI_YH = np.zeros(len(runs),  dtype=object)
    for i, run in enumerate(runs):
        with open(data_path + 'MI_XH_MI_YH_run_{}_{}.pickle'.format(run,ext), 'rb') as f:
            MI_XH, MI_YH = pickle.load(f)
            full_MI_XH[i] = np.array(MI_XH)
            full_MI_YH[i] = np.array(MI_YH)

    avg_MI_XH = np.mean(full_MI_XH, axis = 0)
    avg_MI_YH = np.mean(full_MI_YH, axis = 0)
    plot_info_plan(np.array(avg_MI_XH[:]), np.array(avg_MI_YH[:]), save_plot=save_plot, save_path=save_path)

    return



def bucket_data_for_plots(max_act_values):
    biggest = {}
    smallest = {}

    sorted_max_vals = sorted(max_act_values)
    N = len(sorted_max_vals)
    for i in range(N):
        value = sorted_max_vals[i]
        index = max_act_values.index(value)
        if i < N//2:
            smallest[index] = value
        else:
            biggest[index] = value

    return biggest, smallest