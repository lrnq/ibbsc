import sys
sys.path.insert(0,'../ibbsc')
import numpy as np
import pickle
import plot_utils

RANGE = 1
ext = "256_30adaptive"
data_path = "../data/tanh_adaptive_30_0001/"
# Read in all MI data from different runs
#dims = 
full_MI_XH = np.zeros(RANGE,  dtype=object)
full_MI_YH = np.zeros(RANGE,  dtype=object)
for i in range(RANGE):
	with open(data_path + 'MI_XH_MI_YH_run_{}_{}.pickle'.format(i,ext), 'rb') as f:
		MI_XH, MI_YH = pickle.load(f)
		full_MI_XH[i] = np.array(MI_XH)
		full_MI_YH[i] = np.array(MI_YH)

avg_MI_XH = np.mean(full_MI_XH, axis = 0)
avg_MI_YH = np.mean(full_MI_YH, axis = 0)


plot_utils.plot_layer_MI(avg_MI_XH[:], "I(X;H)")
plot_utils.plot_layer_MI(avg_MI_YH[:], "I(Y;H)")
plot_utils.plot_info_plan(avg_MI_XH[:], avg_MI_YH[:])