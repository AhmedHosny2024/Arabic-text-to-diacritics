from data import DataSet, get_validation
from model import LSTM, train, evaluate_model,GRU
import torch

torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
# Traindata = DataSet( "Dataset/train.txt", batch_size = 64 )
# Traindataloader = Traindata.getdata()
Valdata = DataSet( "Dataset/val.txt", batch_size = 1 )
Validationdataloader = Valdata.getdata()
test = DataSet( "Dataset/test.txt", batch_size = 1 )
testLoader = test.getdata()
# Validationdataloader=get_validation()

inp_vocab_size = 37
hidden_dim = 128
seq_len = 400
num_classes = 15

import pickle
model = LSTM(inp_vocab_size, hidden_dim, seq_len, num_classes)
model.load_state_dict(torch.load('98-4.pth'))
# model = 'LSTM98pickel.pkl'
# with open(model, 'rb') as f:
#     loaded_model = pickle.load(f)
# print("-------------------start training-------------------")
# train(Traindataloader, model)

print("-------------------start evaluating-------------------")
# acc = evaluate_model(Traindataloader, model)
# print("Accuracy: ", acc)

acc = evaluate_model(Validationdataloader, model)
print("Accuracy: ", acc)
acc = evaluate_model(testLoader, model)
print("Accuracy: ", acc)