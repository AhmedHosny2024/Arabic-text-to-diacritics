import pickle as pkl
from torch.utils.data import TensorDataset, DataLoader
import re
import torch
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse
# from gensim.models import KeyedVectors
# from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

import os

# import stanfordnlp

# Download the Arabic models for the neural pipeline
# stanfordnlp.download('ar', force=True)
# Build a neural pipeline using the Arabic models
# nlp = stanfordnlp.Pipeline(lang='ar')

# def split_arabic_sentences_with_stanfordnlp(corpus_text):
#     # Process the text
#     doc = nlp(corpus_text)

#     # Extract sentences from the doc
#     sentences = [sentence.text for sentence in doc.sentences]

#     return sentences
# file_path = '../SG_300_3_400/w2v_SG_300_3_400_10.model'
# word_embed_model = Word2Vec.load(file_path)

max_len=300

with open('/content/drive/MyDrive/NLP project/arabic_letters.pickle', 'rb') as file:
            ARABIC_LETTERS_LIST = pkl.load(file)
with open('/content/drive/MyDrive/NLP project/diacritics.pickle', 'rb') as file:
            DIACRITICS_LIST = pkl.load(file)
with open('/content/drive/MyDrive/NLP project/diacritic2id.pickle', 'rb') as file:
            DIACRITICS_LIST2ID = pkl.load(file)

arabic_letters=[]
for letter in ARABIC_LETTERS_LIST:
    arabic_letters.append(letter[0])
arabic_letters.append(" ")


dicritics=[]
for letter in DIACRITICS_LIST:
    dicritics.append(letter[0])
classes={k:i for i,k in enumerate(DIACRITICS_LIST2ID)}
classes[" "]=15



inverted_classes = {v: k for k, v in classes.items()}

##################### to delete #####################
def read_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def write_to_file_second(dirctory,file_path, text):
    if not os.path.exists(dirctory):
        os.makedirs(dirctory)

    with open(dirctory+"/"+file_path, "w", encoding="utf-8") as file:
        lines=text
        for l in lines:
            file.write(l)
        file.write('\n')
def write_to_file_labels(dirctory,file_path, text):
    if not os.path.exists(dirctory):
        os.makedirs(dirctory)

    with open(dirctory+"/"+file_path, "a", encoding="utf-8") as file:
        lines=text
        for l in lines:
            file.write(inverted_classes[l])
        file.write('\n')


def write_to_file_string(dirctory,file_path, text):
    if not os.path.exists(dirctory):
        os.makedirs(dirctory)

    with open(dirctory+"/"+file_path, "a", encoding="utf-8") as file:
        lines=text
        file.write(lines)
        file.write('\n')
        file.write('\n')
#####################################################

def preprocess(text):
        # Remove URLs
        text = re.sub(r"http[s|S]\S+", "", text,flags=re.MULTILINE)
        text = re.sub(r"www\S+", "", text,flags=re.MULTILINE)
        # Remove English letters
        text = re.sub(r"[A-Za-z]+", "", text,flags=re.MULTILINE)
        # Remove Kashida Arabic character
        text = re.sub(r"\u0640", "", text,flags=re.MULTILINE)
        # Add space before and after the numbers
        text = re.sub(r"(\d+)", r" \1 ", text,flags=re.MULTILINE)
        # removes SHIFT+J Arabic character
        text = re.sub(r"\u0691", "", text,flags=re.MULTILINE)
        # remove english numbers
        text = re.sub(r"[0-9]+", "", text,flags=re.MULTILINE)
        # remove arabic numbers
        text = re.sub(r"[٠-٩]+", "", text,flags=re.MULTILINE)
         # remove brackets
        # text = re.sub(r"\[.*?\]", "", text)
        # text = re.sub(r"\(.*?\)", "", text)
        return text


# text=read_text('Dataset/train.txt')
# text=preprocess(text)
# print(len(text))
# # delete except arabic letters and dicritics and punctuation
# # remove multiple spaces

# write_to_file_string("test","data.txt",text)

def split_text(text):
    text=text.split('.')
    # split text to sentences on all arabic sparatators

    data=[]
    for t in text:
        if(len(t)==0): continue
        if(len(t)<max_len):
            while(len(t)<max_len):
                 t+=" "
            data.append(t)
        if(len(t)>max_len):
            data.append(t[:max_len])
            supdata=t[max_len:]
            while(len(supdata)>max_len):
                data.append(supdata[:max_len])
                supdata=supdata[max_len:]
            if(len(supdata)<max_len):
                while(len(supdata)<max_len):
                    supdata+=" "
                data.append(supdata)
    return data


# [".", "،", ":", "؛", "-", "؟"]
VALID_ARABIC = dicritics + arabic_letters + ['.']

import re

_whitespace_re = re.compile(r"\s+")

def remove_spaces(text):
    text = re.sub(_whitespace_re, " ", text)
    return text


def preprocessing(text):
    text = filter(lambda char: char in VALID_ARABIC, text)
    text = remove_spaces(''.join(list(text)))
    text=text.strip()
    return text



def get_data_labels(text):
    data=""
    labels=[]
    for i in range(len(text)):
        if(text[i] in arabic_letters):
            data+=text[i]
            if(text[i]==" "):
                labels.append(15)
            elif(i+1<len(text) and text[i+1] in dicritics):
                if(i+2<len(text) and text[i+2] in dicritics and (classes.get(text[i+1])==7 or classes.get(text[i+2])==7)):
                    if(text[i+1]==7 ):
                        labels.append(classes[text[i+2]+text[i+1]])
                        i+=2
                    else:
                        labels.append(classes[text[i+1]+text[i+2]])
                        i+=2
                else:
                    labels.append(classes[text[i+1]])
                    i+=1
            else:
                labels.append(14)
    return data,labels

def encoding(text):
    idx=arabic_letters.index(text)
    encode=np.zeros(len(arabic_letters))
    encode[idx]=1
    return torch.tensor(encode,dtype=torch.float32).to(device)


def get_dataloader(encoded_data, encoding_labels,batch_size=1):
    # Create TensorDataset
    dataset = TensorDataset(encoded_data.to(device), encoding_labels.to(device))
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def get_validation():
    text=read_text('Dataset/val.txt')
    text=preprocess(text)
    # size=int(0.02*len(text))
    # text=text[:size]
    # text=split_text(text)
    # write_to_file_second("test","data.txt",text)
    # text=split_text(text)
    data=[]
    labels=[]
    for i in range (len(text)):
        if(text[i] in arabic_letters):
                data+=(text[i])
                if(i+1<len(text) and text[i+1] in dicritics):
                    if(i+2<len(text) and classes[text[i+1]]==4 and text[i+2] in dicritics):
                        labels.append(classes[text[i+1]+text[i+2]])
                        i+=2
                    else:
                        labels.append(classes[text[i+1]])
                        i+=1
                else:
                    labels.append(15)
        # else:
        #     data.append(text[i])
        #     labels.append(15)
    encoded_data = torch.empty(0, len(arabic_letters),dtype=torch.float32)

    for letter in data:
        if letter in arabic_letters:
            x = encoding(letter).unsqueeze(0)
        else:
            x=np.zeros((1,len(arabic_letters)))
            x=torch.tensor(x,dtype=torch.float32)
        encoded_data = torch.cat((encoded_data, x), 0)
    labels=torch.tensor(labels,dtype=torch.long)
    # print(encoded_data.shape)
    # print(labels.shape)
    dataloader=get_dataloader(encoded_data,labels)

    return dataloader

def get_test(path):
    text=read_text(path)
    text=preprocess(text)
    # size=int(0.01*len(text))
    # text=text[:size]
    # print(len(text))
    text = preprocessing(text)
    text="".join(text)
    text=text.split('.')
    data=[]
    labels=[]
    for t in text:
        t=t.strip()
        d=""
        l=[]
        if len(t)>max_len:
            continue
        else:
            d,l=get_data_labels(t)
            if(len(d)==0): continue
            elif(len(d)<max_len):
                data.append(d)
                labels.append(l)
                continue
            elif(len(d)>max_len):
                data.append(d[:max_len])
                labels.append(l[:max_len])
                supdata=d[max_len:]
                suplabels=l[max_len:]
                while(len(supdata)>max_len):
                    data.append(supdata[:max_len])
                    labels.append(suplabels[:max_len])
                    supdata=supdata[max_len:]
                    suplabels=suplabels[max_len:]
                if(len(supdata)<max_len):
                    data.append(supdata)
                    labels.append(suplabels)
    return data,labels

def get_data(path):
    text=read_text(path)
    text=preprocess(text)
    # size=int(len(text))
    # text=text[:size]
    text = preprocessing(text)
    text="".join(text)
    # write_to_file_string("test","data.txt",text)
    # text=split_text(text)
    text=text.split('.')
    print(len(text))
    # print(text[:10])

    # write_to_file_second("test","data.txt",text)
    # # get max length of sentence in text
    # maxdata=text.split('\n')

    # max_len=0
    # for t in maxdata:
    #     if(len(t)>max_len):
    #         max_len=len(t)
    # print(max_len)

    # split text to sentences on all arabic sparatators
    # text = split_arabic_sentences_with_stanfordnlp(text)

    # Filter out empty strings or whitespace-only sentences
    # text = [s.strip() for s in sentences if s.strip()]

    data=[]
    labels=[]
    for t in text:
        d=""
        l=[]
        d,l=get_data_labels(t)
        if(len(d)==0): continue
        if(len(d)<max_len):
            while(len(d)<max_len):
                 d+=" "
                 l.append(15)
            data.append(d)
            labels.append(l)
            continue
        if(len(d)>max_len):
            data.append(d[:max_len])
            labels.append(l[:max_len])
            supdata=d[max_len:]
            suplabels=l[max_len:]
            while(len(supdata)>max_len):
                data.append(supdata[:max_len])
                labels.append(suplabels[:max_len])
                supdata=supdata[max_len:]
                suplabels=suplabels[max_len:]
            if(len(supdata)<max_len):
                while(len(supdata)<max_len):
                    supdata+=" "
                    suplabels.append(15)
                data.append(supdata)
                labels.append(suplabels)
        # data.append(d)
        # labels.append(l)
    print("first sententence :" , data[0])
    # for d in data:
    #     write_to_file_string("test","data.txt",d)
    return data,labels

def get_features(data,labels):
    encoded_data = torch.empty(0, max_len, len(arabic_letters),dtype=torch.float32).to(device)
    for d in data:
        enc = torch.empty(0, len(arabic_letters),dtype=torch.float32).to(device)
        for letter in d:
            x = encoding(letter).unsqueeze(0).to(device)
            enc = torch.cat((enc, x), 0)
        encoded_data = torch.cat((encoded_data, enc.unsqueeze(0)), 0)
    # print(encoded_data.shape)
    encoding_labels=torch.tensor(labels,dtype=torch.long).to(device)
    # print(encoding_labels.shape)
    return encoded_data,encoding_labels

def get_word2vec_features(data, labels, model):
    max_seq_length = 300
    encoded_data = torch.empty(0, max_len, max_seq_length, dtype=torch.float32)

    for sentence in data:
        encoded_sentence = torch.empty(0, max_seq_length, dtype=torch.float32)
        for letter in sentence:
            if letter in model.wv:
                # Convert NumPy array to PyTorch tensor
                 embedding  = torch.tensor(model.wv[letter], dtype=torch.float32).unsqueeze(0)
            else:
                # If the word is not in the model's vocabulary, fill with zeros
                embedding  = torch.zeros((1, max_seq_length), dtype=torch.float32)

            encoded_sentence = torch.cat((encoded_sentence, embedding), 0)

        encoded_data = torch.cat((encoded_data, encoded_sentence.unsqueeze(0)), 0)

    encoding_labels = torch.tensor(labels, dtype=torch.long)

    return encoded_data, encoding_labels


def get_tf_idf_features(data, labels):
    # Create a one-hot encoding tensor with the same dimensions as Word2Vec
    encoded_data = torch.empty(0, max_len, len(arabic_letters), dtype=torch.float32)

    # Create a mapping of words to indices for one-hot encoding
    word_to_idx = {word: idx for idx, word in enumerate(arabic_letters)}

    for sentence in data:
        encoded_sentence = torch.empty(0, len(arabic_letters), dtype=torch.float32)

        for letter in sentence:
            # Use one-hot encoding for each letter
            if letter in word_to_idx:
                idx = word_to_idx[letter]
                one_hot = torch.zeros(len(arabic_letters), dtype=torch.float32)
                one_hot[idx] = 1
                one_hot = one_hot.unsqueeze(0)
            else:
                # If the word is not in the vocabulary, fill with zeros
                one_hot = torch.zeros(len(arabic_letters), dtype=torch.float32).unsqueeze(0)

            encoded_sentence = torch.cat((encoded_sentence, one_hot), 0)

        encoded_data = torch.cat((encoded_data, encoded_sentence.unsqueeze(0)), 0)

    encoding_labels = torch.tensor(labels, dtype=torch.long)

    return encoded_data, encoding_labels



def get_bow_features(data, labels, max_seq_length=400):
    # Create a CountVectorizer
    vectorizer = CountVectorizer(analyzer='char', vocabulary=arabic_letters)

    # Fit and transform the data
    encoded_data = vectorizer.transform(data)

    # Convert to a PyTorch sparse tensor with the specified max_seq_length
    encoded_data = torch.sparse.FloatTensor(
        torch.tensor(encoded_data.nonzero()).t(),
        torch.ones(encoded_data.nnz),
        (len(data), max_seq_length, len(arabic_letters))
    )

    encoding_labels = torch.tensor(labels, dtype=torch.long)

    return encoded_data, encoding_labels

class DataSet():

    def __init__(self,path,batch_size=1,test=False) :
        print("Loading data...")
        if(test):
            data1,labels1=get_test(path)
        else:
            data1,labels1=get_data(path)
        # now labels is list of list [[1,2,3,4,5,15,15,0],[1,2,3,4,5,15,15,0]]
        # data is list of string ['احمد','محمد']
        print("Extracting features...")
        data,labels=get_features(data1,labels1)
        # data,labels=get_word2vec_features(data1,labels1,word_embed_model)
        print(data.shape)
        # now the data and labels are tensor
        # data is tensor of shape (number of sentences,max_len,37)
        # labels is tensor of shape (number of sentences,max_len)
        print("Creating dataloader...")

        dataloader=get_dataloader(data,labels,batch_size)
        self.x=data1
        self.y=labels1
        self.dataloader=dataloader
        print("Done data creation !")

    def __len__(self):
         return len(self.y)
    def item(self,idx):
         return self.x[idx],self.y[idx]
    def getdata(self):
        return self.dataloader
    def getx(self):
        return self.x




