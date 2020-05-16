import plot_utils
import numpy as np
import data_utils


# Define base strings for saved data
num_runs_to_avg = 19
version = "2" # string to indicate that the learning rate is low 
base_ext = "8000[12, 10, 7, 5, 4, 3, 2]tanh"
base_ext_relu2 = "8000[12, 10, 7, 5, 4, 3, 2]reluvariable"
base_ext_relu1 = "8000[12, 10, 7, 5, 4, 3, 2]relu100bins"
size_ext1 = "256"
size_ext2 = "full"

# Tanh paths
data_path_tanh_256 = "data_tanh_256/"
data_path_tanh_full = "data_tanh_full/"

# ReLU paths
data_path_relu_256 = "data_relu_256/"
data_path_relu_full = "data_relu_full/"


# Paths to save in
save_to = "figures/"


#plot_average_MI(num_runs, ext, data_path, save_plot, save_path version="")

# Tanh plots
plot_utils.plot_average_MI(num_runs_to_avg, size_ext1 + base_ext, data_path_tanh_256, True, save_to + "tanh_" + size_ext1)
plot_utils.plot_average_MI(num_runs_to_avg, size_ext2 + base_ext, data_path_tanh_full, True, save_to + "tanh_" + size_ext2)
plot_utils.plot_average_MI(num_runs_to_avg, size_ext1 + base_ext, data_path_tanh_256, True, save_to + "tanh_" + size_ext1 + version, version=version)
plot_utils.plot_average_MI(num_runs_to_avg, size_ext2 + base_ext, data_path_tanh_full, True, save_to + "tanh_" + size_ext2 + version, version=version)

#Relu Plots
plot_utils.plot_average_MI(num_runs_to_avg, size_ext1 + base_ext_relu1, data_path_relu_256, True, save_to + "relu_" + size_ext1)
plot_utils.plot_average_MI(num_runs_to_avg, size_ext2 + base_ext_relu2, data_path_relu_full, True, save_to + "relu_" + size_ext2)
plot_utils.plot_average_MI(num_runs_to_avg, size_ext1 + base_ext_relu2, data_path_relu_256, True, save_to + "relu_" + size_ext1 + version)
plot_utils.plot_average_MI(num_runs_to_avg, size_ext2 + base_ext_relu1, data_path_relu_full, True, save_to + "relu_" + size_ext2 + version)

max_vals_full = [25.632437, 36.014107, 21.595205, 28.120224, 18.14173, 36.99366, 12.874995, 44.74406, 48.047825, 17.537935, 86.94622, 69.85319, 28.477007, 21.444286, 31.934618, 44.490665, 25.073826, 26.558466, 46.902363, 38.84796, 47.31601, 17.688326, 112.087654, 20.4147, 35.948277, 12.132638, 33.50593, 57.990284, 32.06656, 41.895412, 28.17634, 32.225693, 222.41753, 19.171957, 33.966057, 45.325367, 23.702776, 32.391563, 25.885601, 31.805866, 32.460415, 82.32757, 22.395355, 2.2452862, 32.467552, 81.833176, 40.430332, 128.32297, 34.280624, 14.022226]
max_vals_256 = [20.628553, 23.40096, 12.558179, 2.1561344, 51.82339, 25.943825, 38.339085, 43.135033, 23.457582, 78.72383, 43.739132, 24.462122, 36.62319, 13.862715, 12.344459, 25.61324, 12.168928, 16.726175, 46.663143, 28.845692, 31.813265, 15.828995, 63.05508, 24.172821, 19.868414, 15.314455, 30.482254, 23.807362, 63.095123, 24.15069, 21.677345, 53.097794, 37.089592, 51.20914, 20.902552, 27.063944, 23.428076, 11.424805, 26.055439, 13.42807, 9.704986, 42.204105, 12.739449, 2.2452862, 28.483688, 32.078587, 39.14921]

#[20.628553, 23.40096, 12.558179, 2.1561344, 51.82339, 25.943825, 38.339085, 43.135033, 23.457582, 78.72383, 43.739132, 24.462122, 36.62319, 13.862715, 12.344459, 25.61324, 12.168928, 16.726175, 46.663143, 28.845692, 31.813265, 15.828995, 63.05508, 24.172821, 19.868414, 15.314455, 30.482254, 23.807362, 63.095123, 24.15069, 21.677345, 53.097794, 37.089592, 51.20914, 20.902552, 27.063944, 23.428076, 11.424805, 26.055439, 13.42807, 9.704986, 42.204105, 12.739449, 2.2452862, 28.483688, 32.078587, 39.14921, 122.6786, 13.358196, 25.961185]


# Bucket all runs such that the lowest half and the smallest half is collected. 
max_large_full, max_small_full = data_utils.bucket_data_for_plots(max_vals_full)
max_large_256, max_small_256 = data_utils.bucket_data_for_plots(max_vals_256)


#plot_max_vals(max_vals, save_path):
plot_utils.plot_max_vals(max_vals_full[:], "./figures/max_vals_full")
plot_utils.plot_max_vals(max_vals_256[:], "./figures/max_vals_256")


plot_utils.plot_subset_runs_info(max_small_256.keys(), size_ext1 + base_ext_relu1, save_to + "onerelu_" + size_ext1 + "min256", data_path_relu_256)
plot_utils.plot_subset_runs_info(max_large_full.keys(), size_ext2 + base_ext_relu1, save_to + "onerelu_" + size_ext2 + "maxfull", data_path_relu_full)
plot_utils.plot_subset_runs_info(max_large_256.keys(), size_ext1 + base_ext_relu1, save_to + "onerelu_" + size_ext1 + version + "max256", data_path_relu_256)
plot_utils.plot_subset_runs_info(max_small_full.keys(), size_ext2 + base_ext_relu1, save_to + "onerelu_" + size_ext2 + version + "minfull", data_path_relu_full)

