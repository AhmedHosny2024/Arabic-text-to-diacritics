from data import DataSet, get_validation
from model import LSTM, train, evaluate_model
import torch

torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
Traindata = DataSet( "Dataset/train.txt", batch_size = 1 )
Traindataloader = Traindata.getdata()
Validationdataloader=get_validation()

inp_vocab_size = 37
hidden_dim = 64
seq_len = 400
num_classes = 16

model = LSTM(inp_vocab_size, hidden_dim, seq_len, num_classes)

print("-------------------start training-------------------")
train(Traindataloader, model)

print("-------------------start evaluating-------------------")
acc = evaluate_model(Traindataloader, model)
print("Accuracy: ", acc)

acc = evaluate_model(Validationdataloader, model)
print("Accuracy: ", acc)
