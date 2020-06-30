import torch 
from torch import nn, optim
from torch.nn import functional as F
from distributions import truncated_normal_
import tqdm



class Trainer:
    def __init__(self, loss, epochs, model, optimizer, device):
        self.opt = optimizer
        self.device = device
        self.loss_function = loss
        self.epochs = epochs
        self.model = model
        self.hidden_activations = [] # index 1: epoch num, index2 : layer_num
        self.val_loss = []
        self.train_loss = []
        self.full_loss = []
        self.error_train = []
        self.error_test = []
        self.max_value_layers_train = [] # should be of size (num_epochs, depth network)
        self.max_value_layers_mi = [] # should be of size (num_epochs, depth network)
        #self.weights = dict() #  Not currently in use, but if plot of grad of weights are needed we need this
        #self.ws_grads = dict() # Not currently in use, but if plot of grad of weights are needed we need this
        
    def init_weights(self, layer):
        """
        Initialize the weights and bias for each linear layer in the model.
        """
        if type(layer) == nn.Linear:
            # Truncated normal is only available in their unstable nightly version
            #nn.init.trunc_normal_(layer.weight, mean=0, std=1/np.sqrt(layer.weight.shape[0]))
            truncated_normal_(layer.weight)
            if layer.bias != None: 
                layer.bias.data.fill_(0.0)


    def get_max_val(self, activation_values, train=False, mi=False):
        """
        Activation values are a list of size (num_samples, net_depth).
        The function saves the maximum activation values.
        """
        cur_epoch_max = []
        for layer in activation_values:
            cur_epoch_max.append(layer.max())
        if train:
            self.max_value_layers_train.append(cur_epoch_max)
        if mi:
            self.max_value_layers_mi.append(cur_epoch_max)
        return
        
    def evaluate(self, loader, epoch, val=False):
        """
        TODO: This function is poorly named.
        If the val flag is set to true it will pass the test data through the model.
        However no updates are made from the result so hence it should probably
        be called a test flag and not a val flag. 

        If the val flag is true the whole dataset is fed through the model. 

        Args:
            loader: Dataloader object 
            epoch: Int with the current epoch number
            val: flag to indicate if the test data should be passed through the model . Defaults to False.

        Returns:
            v_loss: loss over the data fed through the model
            list(map(lambda x:x.cpu().numpy(), activations)): all activation values 
        """
        self.model.eval()
        v_loss = 0
        acc = 0
        with torch.no_grad(): # Speeds up very little by turning autograd engine off.
            if val:
                for data, label in loader:
                    data, label= data.to(self.device), label.long().to(self.device)
                    yhat, yhat_softmax, activations = self.model(data)
                    v_loss += self.loss_function(yhat, label).item()
                    acc += self.get_number_correct(yhat_softmax, label)
            else:
                data, label = loader.dataset.tensors[0].to(self.device), loader.dataset.tensors[1].long().to(self.device)
                yhat, yhat_softmax, activations = self.model(data)
                v_loss += self.loss_function(yhat, label).item()
                
        v_loss = v_loss / len(loader.dataset)
        if val:
            acc = acc / float(len(loader.dataset))
            self.error_test.append(1-acc)
            if epoch % 100 == 0:
                print('Validation loss: {:.7f},  Validation Acc. {:.4f}'.format(v_loss, acc))
            self.val_loss.append(v_loss)
        else:
            self.full_loss.append(v_loss)
        return v_loss, list(map(lambda x:x.cpu().numpy(), activations))
    
    
    def save_act_loader(self, loader, epoch):
        """
        If we want to save the activity after each epoch for more that one dataset.
        I.e some papers save activity for both train, test and test+train. 
        Note that the order is important here. TODO: change loaders to be a dict.
        """
        _, act = self.evaluate(loader, epoch)
        self.hidden_activations.append(act)
        self.get_max_val(act, mi=True)

    
    def get_number_correct(self, output, target):
        """
        Returns number of correct predictions from the softmax output. 
        Requires target to be a flat vector i.e not one-hot encoded.
        TODO: rewrite
        """
        n_corr = 0
        if output.is_cuda:
            preds = output.argmax(dim=1).cpu().numpy()
        else:
            preds = output.argmax(dim=1).numpy()
        for i in range(len(target)):
            if target[i] == preds[i]:
                n_corr += 1
        return n_corr 
    
    
    def train(self, train_loader, test_loader, act_loader):
        self.model.apply(self.init_weights) #Init kernel weights
        for epoch in range(1, self.epochs+1):
            ### START MAIN TRAIN LOOP ###
            self.model.train()
            train_loss = 0
            acc_train = 0
            for train_data, label in train_loader: 
                train_data, label  = train_data.to(self.device), label.long().to(self.device)
                yhat, yhat_softmax, _ = self.model(train_data)
                loss = self.loss_function(yhat, label)
                acc_train += self.get_number_correct(yhat_softmax, label)
                self.opt.zero_grad()
                loss.backward()
                train_loss += loss.item()
                self.opt.step()
            acc_train = acc_train / float(len(train_loader.dataset))
            self.error_train.append(1-acc_train)
            train_loss = train_loss / len(train_loader.dataset)
            self.train_loss.append(train_loss)
            if epoch % 100 == 0:
                print('Epoch: {} Train loss: {:.7f},  Train Acc. {:.4f}'.format(epoch, train_loss, acc_train))
            ### STOP MAIN TRAIN LOOP ###
        
            ### RUN ON TEST DATA ###
            self.evaluate(test_loader, epoch, val=True)[0]
            ### SAVE ACTIVATION ON FULL DATA ###
            self.save_act_loader(act_loader, epoch)
            if epoch % 100 == 0:
                print("-"*50)
