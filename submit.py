import pandas as pd
import torch
from data import *
from model import *


classes = {
    "َ": 0,
    "ُ": 1,
    "ِ": 2,
    "ْ": 3,
    "ّ": 4,
    "ً": 5,
    "ٌ": 6,
    "ٍ": 7,
    "َّ": 8,
    "ُّ": 9,
    "ِّ": 10,
    "ًّ": 11,
    "ٌّ": 12,
    "ٍّ": 13,
    "": 14,
}

inverted_classes = {v: k for k, v in classes.items()}


inp_vocab_size = 37
hidden_dim = 128
seq_len = 400
num_classes = 15

model = LSTM(inp_vocab_size, hidden_dim, seq_len, num_classes)
model.load_state_dict(torch.load("LSTM98.pth", map_location=torch.device("cpu")))
model.eval()

print("-------------------start evaluating-------------------")

test_sentence = "ذهب علي الى الشاطئ"
print("Test sentence: ", test_sentence)

enc_before = torch.empty(0, inp_vocab_size, dtype=torch.float32)
list_of_letters_before = []
numb_of_lines = 0
for letter in test_sentence:
    if letter == " ":
        continue
    list_of_letters_before.append((numb_of_lines, letter))
    x = encoding(letter).unsqueeze(0)
    enc_before = torch.cat((enc_before, x), 0)
    if letter == "\n":
        numb_of_lines += 1


print("List of arabic letters before predictions: ", list_of_letters_before)
id_line_letter_data_before = [
    (i, line, label) for i, (line, label) in enumerate(list_of_letters_before)
]


print("id, line, letter before predictions: ", id_line_letter_data_before)
id_line_letter_df_before = pd.DataFrame(
    {
        "id": [info[0] for info in id_line_letter_data_before],
        "line": [info[1] for info in id_line_letter_data_before],
        "letter": [info[2] for info in id_line_letter_data_before],
    }
)
id_line_letter_df_before.to_csv("submission_id_line_letter.csv", index=False)


with torch.no_grad():
    yhat = model(enc_before)
    yhat = yhat.detach().cpu().numpy()
    yhat = yhat.reshape(-1, yhat.shape[-1])
    predicted_classes = torch.argmax(torch.from_numpy(yhat), axis=1).tolist()
print("Predicted classes: ", predicted_classes)


result = ""
for i in range(len(predicted_classes)):

    result += test_sentence[i]
    result += inverted_classes[predicted_classes[i]]

print("Reversed string: ", result[::-1])

predicted_classes_csv = [(i, label) for i, label in enumerate(predicted_classes)]

id_line_letter_df_after = pd.DataFrame(
    {
        "id": [info[0] for info in predicted_classes_csv],
        "label": [info[1] for info in predicted_classes_csv],
    }
)
id_line_letter_df_after.to_csv("predicted_chars.csv", index=False)
