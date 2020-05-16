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
        self.hidden_activations_test = []
        self.hidden_activations_train = []
        self.val_loss = []
        self.train_loss = []
        self.full_loss = []
        self.error_train = []
        self.error_test = []
        #self.weights = dict() #  Not currently in use, but if plot of grad of weights are needed we need this
        #self.ws_grads = dict() # Not currently in use, but if plot of grad of weights are needed we need this
        
    def _init_weights(self, layer):
        """
        Initialize the weights and bias for each linear layer in the model.
        """
        if type(layer) == nn.Linear:
            # Truncated normal is only available in their unstable nightly version
            #nn.init.trunc_normal_(layer.weight, mean=0, std=1/np.sqrt(layer.weight.shape[0]))
            truncated_normal_(layer.weight)
            if layer.bias != None: 
                layer.bias.data.fill_(0.00)

        
    def _get_epoch_activity(self, loader, epoch, val=False):
        """
        After each epoch save the activation of each hidden layer
        """
        self.model.eval()
        v_loss = 0
        acc = 0
        with torch.no_grad(): # Speeds up very little by turning autograd engine off.
            for data, label in loader:
                data, label= data.to(self.device), label.long().to(self.device)
                yhat, yhat_softmax, activations = self.model(data)
                v_loss += self.loss_function(yhat, label).item()
                acc += self._get_number_correct(yhat_softmax, label).item()
                
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
    
    
    def _loop_act_loaders(self, loaders, epoch):
        """
        If we want to save the activity after each epoch for more that one dataset.
        I.e some papers save activity for both train, test and test+train. 
        Note that the order is important here. TODO: change loaders to be a dict.
        """
        for counter, loader in enumerate(loaders):
            v_loss, act = self._get_epoch_activity(loader, epoch)
            if counter == 0:
                self.hidden_activations.append(act)
            elif counter == 1:
                self.hidden_activations_train.append(act)
            elif counter == 2:
                self.hidden_activations_test.append(act)
            counter += 1
            
    
    def _get_number_correct(self, output, target):
        """
        Returns number of correct predictions from the softmax output. 
        Requires target to be a flat vector i.e not one-hot encoded.
        """
        n_corr = (target == output.argmax(dim=1)).sum()
        return n_corr 
    
    
    def train(self, train_loader, test_loader, act_loaders):
        MI_tracker = 1
        self.model.apply(self._init_weights) #Init kernel weights
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.opt, 'min', verbose=True, patience=300)
        for epoch in range(1, self.epochs+1):
            ### START MAIN TRAIN LOOP ###
            self.model.train()
            train_loss = 0
            acc_train = 0
            for train_data, label in train_loader: 
                train_data, label  = train_data.to(self.device), label.long().to(self.device)
                yhat, yhat_softmax, _ = self.model(train_data)
                loss = self.loss_function(yhat, label)
                acc_train += self._get_number_correct(yhat_softmax, label).item()
                self.opt.zero_grad()
                loss.backward()
                train_loss += loss.item()
                self.opt.step()
            self.error_train.append(1-acc_train)
            train_loss = train_loss / len(train_loader.dataset)
            self.train_loss.append(train_loss)
            if epoch % 100 == 0:
                print('Epoch: {} Train loss: {:.7f},  Train Acc. {:.4f}'.format(epoch, train_loss, acc_train / float(len(train_loader.dataset))))
            ### STOP MAIN TRAIN LOOP ###
        
            ### RUN ON VALIDATION DATA ###
            val_loss = self._get_epoch_activity(test_loader, epoch, val=True)[0]
            #scheduler.step(val_loss) #Reduce LR on plateau.
            
            ### SAVE ACTIVATION ON FULL DATA ###
            self._loop_act_loaders(act_loaders, epoch)
            if epoch % 100 == 0:
                print("-"*50)