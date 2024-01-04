import torch
import torch.nn as nn

class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim=25, hidden_dim=256, num_layers=2, dropout=0.5):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.blstm1 = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.blstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dense1 = nn.Linear(hidden_dim * 2, 512)
        self.dense2 = nn.Linear(512, 512)
        self.output_layer = nn.Linear(512, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        embedded = self.embedding(x)
        blstm1_out, _ = self.blstm1(embedded)
        dropout1_out = self.dropout1(blstm1_out)
        blstm2_out, _ = self.blstm2(dropout1_out)
        dropout2_out = self.dropout2(blstm2_out)
        dense1_out = self.relu(self.dense1(dropout2_out))
        dense2_out = self.relu(self.dense2(dense1_out))
        output = self.output_layer(dense2_out)
        output = self.softmax(output)
        return output
