from data import DataSet, get_validation
from model import LSTM, train, evaluate_model

Traindata = DataSet("Dataset/train.txt", batch_size=1)
Traindataloader = Traindata.getdata()

inp_vocab_size = 37
hidden_dim = 256
seq_len = 600
num_classes = 19

model = LSTM(inp_vocab_size, hidden_dim, seq_len, num_classes)
print("-------------------start training-------------------")
train(Traindataloader, model)

Validation=DataSet('Dataset/val.txt',batch_size=1)
Validationdataloader=Validation.getdata()
print("-------------------start evaluating-------------------")
acc = evaluate_model(Validationdataloader, model)
print("Accuracy: ", acc)
