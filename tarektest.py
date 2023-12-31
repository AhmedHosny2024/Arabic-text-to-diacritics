import pickle
from data import DataSet
from model import evaluate_model, LSTM

import torch
inp_vocab_size = 37
hidden_dim = 128
seq_len = 400
num_classes = 15

import pickle
model = LSTM(inp_vocab_size, hidden_dim, seq_len, num_classes)
model.load_state_dict(torch.load('LSTM98.pth'))




Valdata=DataSet("Dataset/val.txt",batch_size=1)
Validationdataloader=Valdata.getdata()
acc = evaluate_model(Validationdataloader, model)
print("Accuracy: ", acc)
test = DataSet( "Dataset/test.txt", batch_size = 1 )
testLoader = test.getdata()
acc = evaluate_model(testLoader, model)
print("Accuracy: ", acc)