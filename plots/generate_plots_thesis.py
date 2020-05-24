import sys
sys.path.insert(0,'../ibbsc')
import plot_utils
import numpy as np
import data_utils


num_runs_to_avg = 40

# TODO: Fix hardcoded paths to data


def plot_base_infoplanes():
    ext1 = "256_100bins"
    ext2 = "full_100bins"
    ext3 = "256_30bins"
    ext4 = "full_30bins"

    # ReLU Paths
    data_path_relu_256_fixed_100 = "../data/relu_fixed_/"
    #data_path_relu_full_fixed_100 = "../data/relu_fixed_/"

    # Tanh plots 
    data_path_tanh_256_fixed = "../data/tanh_fixed_/"
    data_path_tanh_full_fixed = "../data/tanh_fixed_/" 

    # Paths to save in
    save_to = "../figures/"

    # Tanh Plots
    plot_utils.plot_average_MI(num_runs_to_avg, ext3, data_path_tanh_256_fixed, True, save_to + "tanh_" + ext3)
    plot_utils.plot_average_MI(num_runs_to_avg, ext4, data_path_tanh_full_fixed, True, save_to + "tanh_" + ext1)

    # ReLU Plots
    plot_utils.plot_average_MI(num_runs_to_avg, ext1, data_path_relu_256_fixed_100, True, save_to + "relu_" + ext1)
    #plot_utils.plot_average_MI(num_runs_to_avg, ext2, data_path_relu_full_fixed_100, True, save_to + "relu_" + ext2)



# Further investigation information plane plots

def plot_further_infoplanes():
    ext1 = "256_30adaptive"
    ext2 = "256_30bins"

    # Tanh paths
    data_path_tanh_256_adap = "../data/tanh_adaptive_30_/"
    #data_path_tanh_full = "data_tanh_fixed_30/" # already plotted in base

    # ReLU paths
    data_path_relu_256_adap = "../data/relu_adaptive_30/"
    data_path_relu_256_fixed = "../data/relu_fixed_/"

    #ELU Paths
    data_path_elu_256_adap = "../data/elu_adaptive_30/"
    data_path_elu_256_fixed = "../data/elu_fixed_30_/"

    # 6ReLU Paths
    data_path_6relu_256_adap = "../data/6relu_adaptive_30/"
    data_path_6relu_256_fixed = "../data/6relu_fixed_30_/"

    # Paths to save in
    save_to = "../figures/"


    #plot_average_MI(num_runs, ext, data_path, save_plot, save_path)

    # Tanh Plot
    plot_utils.plot_average_MI(num_runs_to_avg, ext1, data_path_tanh_256_adap, True, save_to + "tanh_" + ext1)

    # ReLU Plots
    plot_utils.plot_average_MI(num_runs_to_avg, ext1, data_path_relu_256_adap, True, save_to + "relu_" + ext1)
    plot_utils.plot_average_MI(num_runs_to_avg, ext2, data_path_relu_256_fixed, True, save_to + "relu_" + ext2)

    # ELU Plots
    plot_utils.plot_average_MI(num_runs_to_avg, ext1, data_path_elu_256_adap, True, save_to + "elu_" + ext1)
    plot_utils.plot_average_MI(num_runs_to_avg, ext2, data_path_elu_256_fixed, True, save_to + "elu_" + ext2)

    # 6ReLU Plots
    plot_utils.plot_average_MI(num_runs_to_avg, ext1, data_path_6relu_256_adap, True, save_to + "6relu_" + ext1)
    plot_utils.plot_average_MI(num_runs_to_avg, ext2, data_path_6relu_256_fixed, True, save_to + "6relu_" + ext2)