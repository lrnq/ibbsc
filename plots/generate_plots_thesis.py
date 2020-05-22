import sys
sys.path.insert(0,'../ibbsc')
import plot_utils
import numpy as np
import data_utils


# Define base strings for saved data



num_runs_to_avg = 40
ext1 = "256_30adaptive"
ext2 = "256_30bins"



# Tanh paths
data_path_tanh_256_adap = "../data/tanh_adaptive_30_/"
#data_path_tanh_full = "data_tanh_full/"

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
#plot_utils.plot_average_MI(num_runs_to_avg, ext1, data_path_tanh_256_adap, True, save_to + "tanh_" + ext1)

# ReLU Plots
plot_utils.plot_average_MI(num_runs_to_avg, ext1, data_path_relu_256_adap, True, save_to + "relu_" + ext1)
plot_utils.plot_average_MI(num_runs_to_avg, ext2, data_path_relu_256_fixed, True, save_to + "relu_" + ext2)

# ELU Plots
plot_utils.plot_average_MI(num_runs_to_avg, ext1, data_path_elu_256_adap, True, save_to + "elu_" + ext1)
plot_utils.plot_average_MI(num_runs_to_avg, ext2, data_path_elu_256_fixed, True, save_to + "elu_" + ext2)

# 6ReLU Plots
#plot_utils.plot_average_MI(num_runs_to_avg, ext1, data_path_6relu_256_adap, True, save_to + "6relu_" + ext1)
#plot_utils.plot_average_MI(num_runs_to_avg, ext2, data_path_6relu_256_fixed, True, save_to + "6relu_" + ext2)