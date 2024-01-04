import pandas as pd
import torch
from data import *
from model import *
print(device)



inverted_classes = {v: k for k, v in classes.items()}


inp_vocab_size = 37
hidden_dim = 128
seq_len = 400
num_classes = 16

model = LSTM(inp_vocab_size, hidden_dim, seq_len, num_classes)
# model.load_state_dict(torch.load("LSTM98.pth", map_location=torch.device("cpu")))
# model.eval()

Traindata = DataSet( "/content/drive/MyDrive/NLP project/Arabic-text-to-diacritics/Dataset/train.txt", batch_size = 256 )
Traindataloader = Traindata.getdata()

model = LSTM(inp_vocab_size, hidden_dim, seq_len, num_classes)
print("-------------------start training-------------------")
train(Traindataloader, model)


validatindata=DataSet("/content/drive/MyDrive/NLP project/Arabic-text-to-diacritics/Dataset/val.txt",batch_size=1)
validatindataloader=validatindata.getdata()
print("-------------------start evaluating-------------------")


predictions = []
actuals=[]
def evaluate_model(test_dl, model):
    predicted_classes = []
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
        while len(targets)>0 and targets[-1]==15:
            targets=targets[:-1]
            predicted_classes=predicted_classes[:-1]
        # store predictions and actuals
        predictions.extend(predicted_classes)
        actuals.extend(targets.tolist())

data=validatindata.getx()


evaluate_model(validatindataloader,model)
result = ""
for i in range(len(data)):
    for j in range(len(data[i])):
        result += data[i][j]
        result += inverted_classes[predictions[i*10+j]]

# print("Reversed string: ", result[::-1])

# print("Predicted classes: ", predictions)
actuals_res=[]
predictions_res=[]
for i in range(len(actuals)):
  if actuals[i]!=15:
    actuals_res.append(actuals[i])
    predictions_res.append(predictions[i])
predicted_classes_csv = [(i, label) for i, label in enumerate(predictions_res)]

id_line_letter_df_after = pd.DataFrame(
    {
        "id": [info[0] for info in predicted_classes_csv],
        "label": [info[1] for info in predicted_classes_csv],
    }
)
id_line_letter_df_after.to_csv("predicted_chars.csv", index=False)



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

# predictions=[x for x in predictions if x!=15]
# actuals=[x for x in actuals if x!=15]

acc = calculate_DER(np.array(actuals_res), np.array(predictions_res))
print("Accuracy: ", acc)