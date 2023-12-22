from LSTM import LSTM
from TrainTest import *

def read_text(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
def main():
    # read tect from CleanDataset/cleaned_data.txt
    print('----------------start---------------')
    text = read_text('CleanDataset/clean_train.txt')
    without_diacritics=read_text('CleanDataset/clean_without_diacritics.txt')
    print('----------------data readed---------------')
    train_data = []
    test_data = []
    for i in range(100):
        train_data.append((text[i],without_diacritics[i]))
    for i in range(100,120):
        test_data.append((text[i],without_diacritics[i]))
    # create dataloader from text
    train_dl, test_dl=dataloader(train_data,test_data,1)
    print('----------------dataloader created---------------')
    # LSTM model parameters
    inp_vocab_size = 600
    targ_vocab_size = 600
    embedding_dim = 512
    layers_units = [256, 256, 256]
    use_batch_norm = False
    # create LSTM model
    model = LSTM(inp_vocab_size,targ_vocab_size,embedding_dim,layers_units,use_batch_norm)
    train(train_dl,model)
    print('----------------model trained---------------')
    acc=evaluate_model(test_dl,model)
    print("Accuracy: ",acc)
    
    


if __name__ == "__main__":
    main()