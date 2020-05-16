import numpy as np
import torch 
from torch import nn
from torch.nn import functional as F
from scipy import io
from torch.utils.data import DataLoader, random_split, TensorDataset
from sklearn.model_selection import train_test_split

def de_onehot(y_onehot):
    out_arr = []
    for i in y_onehot:
        out_arr.append(np.argmax(i))
    return np.array(out_arr)


def load_data(data_path, test_size):
    # Load data as is
    data = io.loadmat(data_path) # OBS loads in a weird JSON
    X = data["F"] # (4096, 12)
    y = data["y"] # (1, 4096)
    
    # Convert labels to one-hot enc.
    y = y.squeeze() # (4096, )
    #classes = len(np.unique(y))
    #y_onehot = np.eye(classes)[y]
    
    # We use a simple method from sklearn.
    # Original paper uses a custom method, 
    # but it shouldnt matter as long
    # as we shuffle and divide. 
    #X_train, X_test, y_train, y_test = train_test_split(X, y,
    #                                                    test_size=819, # same os orig paper
    #                                                    random_state=1,
    #                                                    shuffle=True,
    #                                                    stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size, # same os orig paper
                                                        shuffle=True,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test


def create_dataloader(X, y, batch_size):
    """
    Expects numpy arrays with data 
    like what is returned by the load_data() 
    function. 
    """
    td = TensorDataset(torch.Tensor(X), torch.Tensor(y))
    return DataLoader(td, batch_size=batch_size, shuffle=True) 