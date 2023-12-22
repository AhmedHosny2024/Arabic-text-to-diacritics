from torch import nn
from torch import optim
import torch
from numpy import vstack
from sklearn.metrics import accuracy_score
# train_data = its a list of tuples (input,target)
def dataloader(train_data, test_data, batch_size):
    # create train dataloader
    train_dl = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # create test dataloader
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_dl, test_dl

# train the model From lab 4
# train dataloader and model
def train(train_dl, model):
    # define the optimization
    criterion = nn.CrossEntropyLoss(ignore_index=0) 
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # enumerate epochs
    for epoch in range(100):
        for i, (inputs, targets) in enumerate(train_dl):
            # convert the input and target to tensor
            inputs = torch.tensor(inputs)
            targets = torch.tensor(targets)
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
            print(f'epoch {epoch} batch {i} loss {loss.item()}')
    # save model to file after training
    torch.save(model.state_dict(), 'model.pth')

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

