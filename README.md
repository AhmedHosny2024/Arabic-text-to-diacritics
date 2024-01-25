<h3 align="center">Arabic Text Diacritization</h3>

<p align="left"> 
  Arabic is one of the most spoken languages around the globe. The same word in the Arabic language
can have different meanings and different pronunciations based on how it is diacritized. so this project aims to predict the diacritics for each character in a given text. 
</p>

## 📝 Table of Contents

- [About](#about)
- [Preprocessing](#preprocessing)
- [Features Extraction](#features-extraction)
- [Training](#training)
- [Testing and Accuracy](#testing-and-accuracy)

## About <a name = "about"></a>
 - we try many implementations for this problem. GRU, one and 3 layers LSTM but we found that the best one is using 3 layers Bi-LSTM.
 - also we try many features like using pre-trained word embedding, tf-idf, and char embedding but we found that the best one is using one hot encoding for each char.
 - it is a classification problem on char-level with 14 classes (no diacritics, fath, damm, ...).

## Preprocessing <a name = "preprocessing"></a>
  - first we remove numbers, english letters, HTML tags, urls using regex.
  - then we remove any char except Arabic letters, diacritics, spaces and (.) .
  - remove extra spaces.
  - split text on (.) .
  - for each char in eaxh sentence remove dicretics from it and make the dicretics is the label. 
  - make each sentence a list of 200 chars and make it the input, if the sentence samller than 200 chars we add padding to it (spaces), if it bigger split it to multiple sentences.
  - create dataloader
## Feature extraction <a name = "features-extraction"></a>
  - we use one hot encoding for each char.

## Training <a name = "training"></a>
  - The main model in the code is a character-level Bidirectional Long Short-Term Memory (BiLSTM) network.
  - The model consists of 3 bidirectional LSTM layer, batch normalization, and an output layer.
  - The LSTM layer is designed to capture contextual information from both forward and backward directions.
  - Batch normalization is applied to normalize hidden states.
  - The output layer produces predictions for diacritic labels.
  - The training loop iterates through epochs, batches, and sequences to train the model.
  - CrossEntropyLoss is used as the loss function, and Adam optimizer for parameter updates.
  - we remove the padding from the output and the label before calculating the loss.

## Testing and Accuracy <a name = "testing-and-accuracy"></a>
  - use the same preprocessing for the test data.
  - load the model and predict the output for each sentence.
  - geneerate csv whcih contain 2 columns, the first one is the char_id and the second one is the predicted diacritics class.
  - we get 96.5% accuracy on the test data.
  
