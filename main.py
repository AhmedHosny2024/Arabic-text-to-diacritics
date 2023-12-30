from data import DataSet, get_validation
from model import LSTM, train, evaluate_model
import torch

torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
Traindata = DataSet( "Dataset/train.txt", batch_size = 256 )
Traindataloader = Traindata.getdata()
Valdata = DataSet( "Dataset/val.txt", batch_size = 1 )
Validationdataloader = Valdata.getdata()

test_data = DataSet("Dataset/test.txt", batch_size=1)
test_dataloader = test_data.getdata()
# Validationdataloader=get_validation()

inp_vocab_size = 300
hidden_dim = 64
seq_len = 400
num_classes = 16

model = LSTM(inp_vocab_size, hidden_dim, seq_len, num_classes)

print("-------------------start training-------------------")
train(Traindataloader, model)

print("-------------------start evaluating-------------------")
acc = evaluate_model(Validationdataloader, model)
print("Accuracy: ", acc)

acc = evaluate_model(test_dataloader, model)
print("Accuracy: ", acc)
