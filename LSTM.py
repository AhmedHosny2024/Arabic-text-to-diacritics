from numpy import vstack
from sklearn.base import accuracy_score
from torch import nn
import torch
from torch import optim

class LSTM(nn.Module):
    def __init__( self,
        inp_vocab_size: int,targ_vocab_size: int,embedding_dim: int = 512,
        layers_units: list[int] = [256, 256, 256],use_batch_norm: bool = False):
        super().__init__()
        self.target_vocab_size = targ_vocab_size
        self.embedding = nn.Embedding(inp_vocab_size, embedding_dim)
        # create LSTM layers numer = layers_units
        layers_units = [embedding_dim//2] + layers_units
        layers=[]
        for i in range(1,len(layers_units)):
            layers.append(nn.LSTM(layers_units[i-1]*2,layers_units[i],
                                  batch_first=True,bidirectional=True))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(layers_units[i]))
        self.layers=nn.ModuleList(layers)
        self.projection = nn.Linear(layers_units[-1]*2, targ_vocab_size)
        self.layers_units = layers_units
        self.use_batch_norm = use_batch_norm

    def forward(self, text: torch.Tensor):
        output = self.embedding(text)
        for i , layer in enumerate(self.layers):
            if(isinstance(layer,nn.BatchNorm1d)):
                output = output.permute(0,2,1)
                output = layer(output)
                output = output.permute(0,2,1)
                continue
            if i>0:
                output,(hn,cn)=layer(output,(hn,cn))
            else:
                output,(hn,cn)=layer(output)
        output = self.projection(output)
        return output
      
# train the model From lab 4
# train dataloader and model
def train(train_dl, model):
    # define the optimization
    criterion = nn.CrossEntropyLoss(ignore_index=0) 
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # enumerate epochs
    for epoch in range(100):
        # enumerate mini batches
        # the input should be the arabic text without the diacritics
        # the target should be the arabic text with the diacritics
        # the input and target should be with shape (batch_size,seq_len)
        # train_dl is the train dataloader with shape (batch_size,seq_len)
        # batch_size is the number of samples in each batch
        # seq_len is the length of each sample
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()

# evaluate the model From lab 4
# test dataloader and model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc


        
    
 