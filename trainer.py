import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import numpy as np


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

        self._optim.zero_grad() # Reset gradients
        x = self._model.forward(x) # Propagate
        calc_loss = self._crit(x,y) # The calculated loss
        calc_loss.backward() # Compute gradients
        self._optim.step() # does the update

        return calc_loss
        
    
    def val_test_step(self, x, y):
        
        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        prediction = self._model.forward(x)
        calc_loss = self._crit(prediction, y.float())
        
        return calc_loss
        
    def train_epoch(self):
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it

        loss_list = [] # This is a list that keeps all the losses for the epoch
        self._model.train() # set training mode

        for image, labels in self._train_dl: # For each image/labels couple in the training dataset
            if self._cuda:
                image, labels = image.cuda(), labels.cuda()
            train_loss = self.train_step(image, labels)
            loss_list.append(train_loss)

        avg_loss = sum(loss_list)/len(loss_list) # calculate the average loss for the epoch and return it
        print(f"Average training loss: {avg_loss}")
        return avg_loss


    def val_test(self): 
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics

        self._model.eval() # Set eval mode
        t.no_grad() # disable gradient computation # Since you don't need to update the weights during testing, gradients aren't required anymore. 
        loss_list = []
        for image, labels in self._val_test_dl:
            if self._cuda:
                image, labels = image.cuda(), labels.cuda()
            val_loss = self.val_test_step(image, labels)
            loss_list.append(val_loss)

        avg_loss = sum(loss_list)/len(loss_list) # Calculate the average loss
        print(f"Average validation loss: {avg_loss}")
        return avg_loss
        
    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        train_losses = []
        val_losses = []
        epoch_counter = 0

        es_check = 0 # Early stopping check: will count the times the validation loss was not decreasing
        previous_loss = self.val_test()

        while True:
            # stop by epoch number # TO discuss
            # if epoch_counter > epochs:
            #     break

            # train for an epoch and then calculate the loss and metrics on the validation set
            train_loss = self.train_epoch()
            val_loss = self.val_test()

            train_losses.append(train_loss)
            val_losses.append(val_loss)  # append the losses to the respective lists
           
            self.save_checkpoint(epoch_counter) # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)

            #TODO : early stopping
            if val_loss > previous_loss:
                es_check += 1 # Count how many times the validation loss does NOT increase
            
            if es_check >= self._early_stopping_patience:
                print('Patience exceeded. Early stopping.')
                break

            epoch_counter += 1
            print(f"Epoch: {epoch_counter}, Train_loss: {train_loss}, Val_loss: {val_loss}")

        return train_losses, val_losses
