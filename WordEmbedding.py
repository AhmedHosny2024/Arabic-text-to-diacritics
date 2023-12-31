import torch
import torch.nn as nn
import torch.optim as optim
from data import DataSet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data=DataSet("Dataset/train.txt",batch_size=1)
corpus,labels=data.get_x_y()

# Create word-to-index and index-to-word dictionaries
word_to_ix = {}
for sentence in corpus:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

vocab_size = len(word_to_ix)
embedding_dim = 100  # Dimensionality of the word embeddings

class WordEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(WordEmbeddingModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        return embeds

# Convert words to indices in the corpus
indexed_corpus = []
for sentence in corpus:
    indexed_sentence = [word_to_ix[word] for word in sentence]
    indexed_corpus.append(torch.tensor(indexed_sentence, dtype=torch.long).to(device))

model = WordEmbeddingModel(vocab_size, embedding_dim)
model.to(device)
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


# Training loop
for epoch in range(100):
    total_loss = 0
    for sentence in indexed_corpus:
        optimizer.zero_grad()

        inputs = sentence[:-1]  # Using words as input
        targets = sentence[1:]  # Predicting the next word
        print(inputs.shape)
        print(targets.shape)
        output = model(inputs)
        print(output.shape)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss}")
# Save the model
torch.save(model.state_dict(), 'word_embedding.pth')

# Load the model
# model = WordEmbeddingModel(vocab_size, embedding_dim)
# model.load_state_dict(torch.load('word_embedding.pth'))
# test model
# test_sentence = "مدخلش من الاول فاكيد مش فاهم اللي بيحصل دلوقتي هيتوه"
# test_sentence = test_sentence.split()
# indexed_test_sentence = [word_to_ix[word] for word in test_sentence]
# inputs = torch.tensor(indexed_test_sentence[:-1], dtype=torch.long).to(device)
# targets = torch.tensor(indexed_test_sentence[1:], dtype=torch.long).to(device)
# output = model(inputs)


