import numpy as np
import pickle
import info_utils
import tqdm
from custom_exceptions import MIMethodError


class MI:
    def __init__(self, activity, data_loader, act, num_of_bins):
        self.activity = activity
        self.num_of_bins = num_of_bins
        self.act=act
        self.min_val, self.max_val = info_utils.get_min_max_vals(act, activity)
        
        self.X = data_loader.dataset.tensors[0].numpy()
        self.y_flat = data_loader.dataset.tensors[1].numpy()
        
        classes = len(np.unique(self.y_flat))
        self.y_onehot = np.eye(classes)[self.y_flat.astype("int")]
        nb_classes = self.y_onehot[0]
        self.y_idx_label = {x : None for x in nb_classes}
        for i in self.y_idx_label:
            self.y_idx_label[i] = i == self.y_flat

        patterns, inv, cnts = np.unique(self.X, axis=0, return_inverse=True, return_counts=True)
        self.x_idx_patterns = {x : None for x in range(len(patterns))}
        for i in range(len(patterns)):
            self.x_idx_patterns[i] = i == inv


    def get_prob_dist(self, bins, activations):
        binned = np.digitize(activations, bins)
        _, unique_layers = np.unique(binned, axis=0, return_counts=True)
        prob_hidden_layers = unique_layers / sum(unique_layers)
        return prob_hidden_layers


    def entropy(self, bins, activations):
        prob_hidden_layers = self.get_prob_dist(bins, activations)
        return -np.sum(prob_hidden_layers * np.log2(prob_hidden_layers))


    def cond_entropy(self, cond_bool, activations_layer, bins):
        return sum([self.entropy(bins, activations_layer[inds,:]) * inds.mean() for inds in cond_bool.values()])

    
    def mutual_information(self, cond_bool, activations_layer, bins):
        entropy_layer = self.entropy(bins, activations_layer) # H(h)
        entropy_layer_output = self.cond_entropy(cond_bool, activations_layer, bins) # \sum_y Pr[Y=y] * H(h|Y=y)
        # Note that H(h|X) = 0 so I(X;h) = H(h). Thus the below is not needed.
        #entropy_layer_input = sum([self.entropy(bins, activations_layer[inds,:]) * inds.mean() for inds in self.x_idx_patterns.values()])
        return entropy_layer, (entropy_layer - entropy_layer_output)

    
    def get_mi(self, method):
        all_MI_XH = [] # Contains I(X;H) and stores it as (epoch_num, layer_num)
        all_MI_YH = [] # Contains I(Y;H) and stores it as (epoch_num, layer_num
        if method == "fixed":
            bins = np.linspace(self.min_val, self.max_val, self.num_of_bins)
            print(self.min_val, self.max_val, self.num_of_bins)
        elif method == "adaptive":
            adapt_bins = info_utils.get_bins_layers(self.activity, self.num_of_bins, self.act)
            #with open('bins.pickle', 'wb') as f:
            #        pickle.dump(adapt_bins, f, protocol=pickle.HIGHEST_PROTOCOL)
            #        f.close()
        else:
            raise MIMethodError("Method not supported. Pick fixed or adaptive")

        for idx, epoch in tqdm.tqdm(enumerate(self.activity)):
            temp_MI_XH = []
            temp_MI_YH = []
            for layer_num in range(len(epoch)):
                if method == "fixed":
                    MI_XH, MI_YH = self.mutual_information(self.y_idx_label,epoch[layer_num], bins)
                elif method == "adaptive":
                    MI_XH, MI_YH = self.mutual_information(self.y_idx_label,epoch[layer_num], adapt_bins[idx][layer_num])
                temp_MI_XH.append(MI_XH)
                temp_MI_YH.append(MI_YH)
            all_MI_XH.append(temp_MI_XH)
            all_MI_YH.append(temp_MI_YH)

        return all_MI_XH, all_MI_YH