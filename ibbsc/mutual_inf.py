import numpy as np
import info_utils
import tqdm


class MI:
    def __init__(self, activity, data_loader, act, num_of_bins):
        self.activity = activity
        self.num_of_bins = num_of_bins
        self.act=act
        
        if act == "tanh" or act == "elu":
            self.min_val = -1
            self.max_val = 1
        else:
            self.min_val = 0
            self.max_val = info_utils.get_max_value(activity)
        
        self.X = data_loader.dataset.tensors[0].numpy()
        self.y_flat = data_loader.dataset.tensors[1].numpy()
        
        classes = len(np.unique(self.y_flat))
        self.y_onehot = np.eye(classes)[self.y_flat.astype("int")]
        nb_classes = self.y_onehot[0]
        self.y_idx_label = {x : None for x in nb_classes}
        for i in self.y_idx_label:
            self.y_idx_label[i] = i == self.y_flat


    def entropy(self, bins, activations):
        binned = np.digitize(activations, bins)
        _, unique_layers = np.unique(binned, axis=0, return_counts=True)
        prob_hidden_layers = unique_layers / sum(unique_layers)
        return -np.sum(prob_hidden_layers * np.log2(prob_hidden_layers))

    
    def mi_binning(self, labelixs, activations_layer, bins):
        # H(h). Note that H(h|X) = 0 so I(X;h) = H(h)
        entropy_layer = self.entropy(bins, activations_layer)
        # below is \sum_y Pr[Y=y] * H(h|Y=y)
        entropy_layer_output = sum([self.entropy(bins, activations_layer[inds]) * inds.mean() for inds in labelixs.values()])
        return entropy_layer, (entropy_layer - entropy_layer_output)

    
    def get_MI(self, method):
        all_MI_XH = [] # Contains I(X;H) and stores it as (epoch_num, layer_num)
        all_MI_YH = [] # Contains I(Y;H) and stores it as (epoch_num, layer_num
        if method == "fixed":
            bins = np.linspace(self.min_val, self.max_val, self.num_of_bins)
        elif method == "adaptive":
            adapt_bins = info_utils.get_bins_layers(self.activity, self.num_of_bins, self.act)
        else:
            raise("Method not supported. Pick fixed or adaptive")

        for idx, epoch in tqdm.tqdm(enumerate(self.activity)):
            temp_MI_XH = []
            temp_MI_YH = []
            for layer_num in range(len(epoch)):
                if method == "fixed":
                    MI_XH, MI_YH = self.mi_binning(self.y_idx_label,epoch[layer_num], bins)
                elif method == "adaptive":
                    MI_XH, MI_YH = self.mi_binning(self.y_idx_label,epoch[layer_num], adapt_bins[idx][layer_num])
                temp_MI_XH.append(MI_XH)
                temp_MI_YH.append(MI_YH)
            all_MI_XH.append(temp_MI_XH)
            all_MI_YH.append(temp_MI_YH)

        return all_MI_XH, all_MI_YH





