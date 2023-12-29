import pickle as pkl

with open('files/arabic_letters.pickle', 'rb') as file:
            ARABIC_LETTERS_LIST = pkl.load(file)
with open('files/diacritics.pickle', 'rb') as file:
            DIACRITICS_LIST = pkl.load(file)
  
arabic_letters=[]
for letter in ARABIC_LETTERS_LIST:
    arabic_letters.append(letter[0])
arabic_letters.append(" ")

dicritics=[]
for letter in DIACRITICS_LIST:
    dicritics.append(letter[0])

classes = {
    'َ': 0,
    'ُ': 1,
    'ِ': 2,
    'ْ': 3,
    'ّ': 4,
    'ً': 5,
    'ٌ': 6,
    'ٍ': 7,
    'َّ': 8,
    'ُّ': 9,
    'ِّ': 10,
    'ًّ': 11,
    'ٌّ': 12,
    'ٍّ': 13,
    'ّّ': 14,
    "":15,
}

inverted_classes = {v: k for k, v in classes.items()}

##################### to delete #####################
import os
def read_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()
    
def write_to_file_second(dirctory,file_path, text):
    if not os.path.exists(dirctory):
        os.makedirs(dirctory)

    with open(dirctory+"/"+file_path, "a", encoding="utf-8") as file:
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
#####################################################
import re
import torch
import numpy as np    
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
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"\(.*?\)", "", text)
        return text

def split_text(text):
    text=text.split('.')
    data=[]
    for t in text:
        if(len(t)==0): continue
        if(len(t)<600):
            while(len(t)<600):
                 t+=" "
            data.append(t)
        if(len(t)>600):
             data.append(t[:600])
    return data

def get_data_labels(text):
    data=""
    labels=[]
    for i in range(len(text)):
            if(text[i] in arabic_letters and text[i]):
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
    return data,labels

def one_hot_encoding(text):
    onehot_encoded=[]
    for i in range(len(text)):
         if text[i] in arabic_letters:
            idx=arabic_letters.index(text[i])
            encode=np.zeros(len(arabic_letters))
            encode[idx]=1
            onehot_encoded.append(encode)
    onehot_encoded=torch.tensor(onehot_encoded)
    return onehot_encoded 

def encoding(text):
    idx=arabic_letters.index(text)
    encode=np.zeros(len(arabic_letters))
    encode[idx]=1
    return torch.tensor(encode,dtype=torch.float32)

# text=read_text("Dataset/train.txt")
# text=preprocess(text)
# text=split_text(text)
# text=text[:20]
# data=[]
# labels=[]
# for t in text:
#     d=""
#     l=[]
#     d,l=get_data_labels(t)
#     if(len(d)<600):
#         while(len(d)<600):
#             d+=(" ")
#             l.append(15)
#     else:
#         d=d[:600]
#         l=l[:600]
#     data.append(d)
#     labels.append(l)

# # features extraction
# encoded_data = torch.empty(0, 600, len(arabic_letters))
# for d in data:
#     # check error
#     if(len(d)!=600):
#          print(len(d))
#          print("#"*10)
    
#     enc = torch.empty(0, len(arabic_letters))
#     for letter in d:
#         x = encoding(letter).unsqueeze(0)
#         enc = torch.cat((enc, x), 0)
#     encoded_data = torch.cat((encoded_data, enc.unsqueeze(0)), 0)
# print(encoded_data.shape)
# encoding_labels=torch.tensor(labels)
# print(encoding_labels.shape)

# for i in range(len(data)):
#     write_to_file_second("test","data.txt",data[i])
#     write_to_file_labels("test","data.txt",labels[i])    
# print(len(data))
# print(len(labels))

# for letter in arabic_letters:
#   write_to_file_second("test","data.txt",arabic_letters[len(arabic_letters)-1])

# print(encoding('ا'))
# print("-------------------")
# text=read_text("Dataset/train.txt")
# text=preprocess(text)
# text=split_text(text)
# print(len(text))

# data,labels=get_data_labels(text)
# print(type(data[0]))
# print(labels[0])
from torch.utils.data import TensorDataset, DataLoader

def get_dataloader(encoded_data, encoding_labels,batch_size=1):
# Create TensorDataset
    dataset = TensorDataset(encoded_data, encoding_labels)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def get_validation():
    text=read_text('Dataset/val.txt')
    # text=text[:100]
    text=preprocess(text)
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
        else:
            data.append(text[i])
            labels.append(15)
    encoded_data = torch.empty(0, len(arabic_letters),dtype=torch.float32)
    
    for letter in data:
        if letter in arabic_letters:
            x = encoding(letter).unsqueeze(0)
        else:
            x=np.zeros((1,len(arabic_letters)))
            x=torch.tensor(x,dtype=torch.float32)
        encoded_data = torch.cat((encoded_data, x), 0)
    labels=torch.tensor(labels,dtype=torch.long)
    print(encoded_data.shape)
    print(labels.shape)
    dataloader=get_dataloader(encoded_data,labels)
    
    return dataloader

def get_data(path):
    text=read_text(path)
    text=text[:200]
    text=preprocess(text)
    text=split_text(text)
    data=[]
    labels=[]
    for t in text:
        d=""
        l=[]
        d,l=get_data_labels(t)
        if(len(d)<600):
            while(len(d)<600):
                d+=(" ")
                l.append(15)
        else:
            d=d[:600]
            l=l[:600]
        data.append(d)
        labels.append(l)
    return data,labels

def get_features(data,labels):
    encoded_data = torch.empty(0, 600, len(arabic_letters),dtype=torch.float32)
    for d in data:
        enc = torch.empty(0, len(arabic_letters),dtype=torch.float32)
        for letter in d:
            x = encoding(letter).unsqueeze(0)
            enc = torch.cat((enc, x), 0)
        encoded_data = torch.cat((encoded_data, enc.unsqueeze(0)), 0)
    print(encoded_data.shape)
    encoding_labels=torch.tensor(labels,dtype=torch.long)
    print(encoding_labels.shape)
    return encoded_data,encoding_labels

class DataSet():

    def __init__(self,path,batch_size=1) :
        print("Loading data...")
        data,labels=get_data(path)
        # now labels is list of list [[1,2,3,4,5,15,15,0],[1,2,3,4,5,15,15,0]]
        # data is list of string ['احمد','محمد']
        print("Extracting features...")
        data,labels=get_features(data,labels)
        # now the data and labels are tensor
        # data is tensor of shape (number of sentences,600,37)
        # labels is tensor of shape (number of sentences,600)
        print("Creating dataloader...")
        dataloader=get_dataloader(data,labels,batch_size)
        self.x=data
        self.y=labels
        self.dataloader=dataloader
        print("Done data creation !")

    def __len__(self):
         return len(self.y)
    def item(self,idx):
         return encoding(self.x[idx]),self.y[idx]
    def getdata(self):
        return self.dataloader
    



        