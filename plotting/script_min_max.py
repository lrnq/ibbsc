from collections import Counter
import sys
sys.path.insert(0,'../ibbsc')
import numpy as np
import pickle
import plot_utils


max_vals_256 = [26.156553, 11.790233, 13.940286, 26.172419, 16.100565, 31.788008, 101.462135, 31.9795, 20.14871, 219.29933, 42.236473, 26.416883, 53.83367, 29.040707, 12.553174, 12.181808, 14.545705, 64.85367, 37.27849, 24.101885, 53.01453, 15.266565, 67.228226, 64.52185, 18.337048, 11.995878, 43.305008, 38.763176, 34.29926, 30.414415, 52.64275, 52.454205, 66.02126, 18.143549, 22.604807, 19.741474, 23.865791, 11.856256, 33.452232, 13.452362]
max_cnts = Counter(max_vals_256[:])

b, s = plot_utils.bucket_data_for_plots(max_vals_256)

RANGE = 20
ext = "256_100bins"
data_path = "../data/saved_relu100/"
# Read in all MI data from different runs
full_MI_XH = np.zeros(RANGE,  dtype=object)
full_MI_YH = np.zeros(RANGE,  dtype=object)
for i, idx in enumerate(s):
    with open(data_path + 'MI_XH_MI_YH_run_{}_{}.pickle'.format(idx,ext), 'rb') as f:
        MI_XH, MI_YH = pickle.load(f)
        full_MI_XH[i] = np.array(MI_XH)
        full_MI_YH[i] = np.array(MI_YH)

avg_MI_XH = np.mean(full_MI_XH, axis = 0)
avg_MI_YH = np.mean(full_MI_YH, axis = 0)


plot_utils.plot_layer_MI(avg_MI_XH[:], "$I(X;T)$", save_path="xtrelusmall_" + ext + ".png")
plot_utils.plot_layer_MI(avg_MI_YH[:], "$I(Y;T)$", save_path="ytrelusmall_" + ext + ".png")
plot_utils.plot_info_plan(avg_MI_XH[:], avg_MI_YH[:], save_plot=True, save_path="relusmall_" + ext + ".png")