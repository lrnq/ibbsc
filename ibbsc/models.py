import torch 
from torch import nn
from torch.nn import functional as F



class FNN(nn.Module):
    def __init__(self, layer_sizes, activation="tanh", seed=0):
        super(FNN, self).__init__()
        torch.manual_seed(seed)
        self.linears = nn.ModuleList()
        self.activation = activation
        self.num_layers = len(layer_sizes)
        self.inp_size = layer_sizes[0]
        
        h_in = self.inp_size
        for h_out in layer_sizes[1:]:
            self.linears.append(nn.Linear(h_in, h_out))
            h_in = h_out
    
        
    def forward(self, x):
        activations = [] #TODO: Could be nicer
        for idx in range(self.num_layers-2):
            x = self.linears[idx](x)
            # TODO: Maybe just pass the actual function to the constructor.
            # However this also restrict it to the activation function that
            # the mutual information estimation is supported of currently.
            if self.activation == "tanh":
                x = torch.tanh(x)
            elif self.activation == "relu":
                x = F.relu(x)
            elif self.activation == "elu":
                x = F.elu(x)
            elif self.activation == "6relu":
                x = F.relu6(x)
            else:
                raise("Activation Function not supported...")
            if not self.training: #Internal flag in model
                activations.append(x)
        x = self.linears[-1](x)
        if not self.training: #Internal flag in model
            activations.append(F.softmax(x, dim=-1)) # Cross entropy loss in pytorch adds log(softmax(x)) 
            
        return x, F.softmax(x, dim=-1), activations # Currently only supports multiclass outputs as this softmax is hardcoded in.