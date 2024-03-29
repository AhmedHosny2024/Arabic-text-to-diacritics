import torch
import torch.nn as nn

from torch import optim
from numpy import vstack
import numpy as np

import os
def write_to_file_string(dirctory,file_path, text):
    if not os.path.exists(dirctory):
        os.makedirs(dirctory)

    with open(dirctory+"/"+file_path, "a", encoding="utf-8") as file:
        lines=text
        file.write(lines)
        file.write('\n')

class LSTM(nn.Module):
    def __init__(self, inp_vocab_size: int, hidden_dim: int = 256, seq_len: int = 600, num_classes: int = 16):
        super().__init__()
        self.lstm = nn.LSTM(inp_vocab_size, hidden_dim,num_layers=3, batch_first=True, bidirectional=True)
        # self.lstm = nn.LSTM(inp_vocab_size, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # Output layer for 0 to 16 integers

    def forward(self, input_sequence: torch.Tensor):
        output, _ = self.lstm(input_sequence)
        output = self.fc(output)
        return output

class GRU(nn.Module):
    def __init__(self, inp_vocab_size: int, hidden_dim: int = 256, seq_len: int = 600, num_classes: int = 16):
        super().__init__()
        self.gru = nn.GRU(inp_vocab_size, hidden_dim,num_layers=3, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # Output layer for 0 to 16 integers

    def forward(self, input_sequence: torch.Tensor):
        output, _ = self.gru(input_sequence)
        output = self.fc(output)
        return output


def train(train_dl, model):
    # define the optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.to(device)
    # enumerate epochs
    for epoch in range(20):
        for i, (inputs, targets) in enumerate(train_dl):
            # convert the input and target to tensor
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs.to(device))
            yhat = yhat.view(-1, yhat.size(2))  # Reshape model output to [batch_size * sequence_length, num_classes]
            targets = targets.view(-1)  # Reshape targets to [batch_size * sequence_length]
            # calculate loss
            while len(targets)>0 and targets[-1]==15:
              targets=targets[:-1]
              yhat=yhat[:-1]
            loss = criterion(yhat, targets.to(device))
            loss.backward()
            # update model weights
            optimizer.step()
            print(f'epoch {epoch} batch {i} loss {loss.item()}')
            # break
    # save model to file after training
    torch.save(model.state_dict(), 'model.pth')


def calculate_DER(actual_labels, predicted_labels):
    # Convert lists to PyTorch tensors if they are not already
    if not isinstance(actual_labels, torch.Tensor):
        actual_labels = torch.tensor(actual_labels)
    if not isinstance(predicted_labels, torch.Tensor):
        predicted_labels = torch.tensor(predicted_labels)

    # Check if the lengths of both label sequences match
    if len(actual_labels) != len(predicted_labels):
        raise ValueError("Lengths of actual and predicted labels should match.")

    total_errors = torch.sum(actual_labels != predicted_labels)
    total_frames = len(actual_labels)

    # DER calculation
    DER = (1-(total_errors / total_frames)) * 100.0
    return DER.item()  # Convert PyTorch scalar to Python float


def evaluate_model(test_dl, model):
    predictions, actuals = [], []
    model.to(device)
    for i, (inputs, targets) in enumerate(test_dl):
        yhat = model(inputs.to(device))
        yhat = yhat.detach().cpu().numpy()
        # reshape the outputs to [batch_size * sequence_length, num_classes]
        yhat = yhat.reshape(-1, yhat.shape[-1])
        # get predicted classes
        predicted_classes = np.argmax(yhat, axis=1)
        # convert targets to numpy array and reshape
        targets = targets.cpu().numpy().reshape(-1)
        # store predictions and actuals
        predictions.extend(predicted_classes.tolist())
        actuals.extend(targets.tolist())
        # break
    # calculate accuracy
    acc = calculate_DER(np.array(actuals), np.array(predictions))

    return acc
