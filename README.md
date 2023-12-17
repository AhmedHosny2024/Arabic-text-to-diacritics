# Arabic-text-to-diacritics

### Tasks

### Data preprocessing (Cleaning , Tokenization) Tarek

input = train data file (from eng)
output = text after preprocessing (with diacritics [golden output] , without diacritcs  )

### Features ( at least 3 ) sherif

without diacritcs file

Bag of Words
TF-IDF
Word Embeddings
Trainable embeddings

first element in the list is represent the first line in the with diacritics [golden output]
length of the output feature list == number of with diacritics [golden output] lines

### models Hosny
input = with diacritics [golden output] , without diacritcs

make dataloader and send it to train function
RNN
LSTM
GRU

output = all dicretics text in output file