from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import gensim
import multiprocessing
import numpy as np


# import spacy
import numpy as np
from nltk.tokenize import word_tokenize


# from nltk import pos_tag,ne_chunk
class ArabicTextFeatures:
    def __init__(self, text):
        # load model for e3rab each word and extract meaning of each word
        # self.nlp = spacy.load("xx_sent_ud_sm")
        # self.doc=self.nlp(text)
        self.text = text
        print("Arabic Text Features Initialized")

    def bag_of_words(self):
        # output = first element in the list is represent the first line
        # in the with diacritics [golden output] length of the output feature list == number of with diacritics [golden output] lines
        vectorizer = CountVectorizer()
        list_of_bow = []
        for line in self.text.split("\n"):
            bow = vectorizer.fit_transform([line])
            list_of_bow.append(bow.toarray())
        return list_of_bow

    def tfidf(self):
        vectorizer = TfidfVectorizer(stop_words=None)
        list_of_tfidf = []
        for line in self.text.split("\n"):
            if line.strip():
                try:
                    tfidf = vectorizer.fit_transform([line])
                    list_of_tfidf.append(tfidf.toarray())
                except ValueError:
                    list_of_tfidf.append(None)
            else:
                list_of_tfidf.append(None)
        return list_of_tfidf

    def char_ngrams(self, n=2):
        ngrams = [self.text[i : i + n] for i in range(len(self.text) - n + 1)]
        return ngrams

    # todo implement word embeddings

    def word_embeddings(self, model):
        list_of_embeddings = []
        lines = self.text.split("\n")
        print("Number of lines to process:", len(lines)) 

        for idx, line in enumerate(lines):
            print(f"Processing line {idx + 1}: {line}") 
            words = line.split()
            print("Number of words in line:", len(words))

            embeddings = []
            for word in words:
                if word in model.wv:
                    embeddings.append(model.wv[word])
                else:
                    embeddings.append(np.zeros(model.vector_size))
            list_of_embeddings.append(embeddings)
        return list_of_embeddings





# model = Word2Vec(sentences, min_count=1)





    # part of speech tagging
    def POS(self):
        # apply pos_ method on each token in the text return token.pos_ for each token except X(unknown word), SPACE(whitespace)
        # pos = [token.pos_ for token in self.doc if(token.pos_ not in ["X","SPACE"])]

        # implement POS using nltk
        # tokenized_text = word_tokenize(self.text)
        # pos = pos_tag(tokenized_text, lang='ar')
        return ""

    # named entity recognition
    # def NER(self):
    #     # apply ent_type_ method on each token in the text return token.ent_type_ for each token except X(unknown word), SPACE(whitespace)

    #     print('----------------NER0000000----------------')
    #     print(self.text)
    #     print('----------------NER0000000----------------')
    #     processed_text = self.text.split('\n')
    #     print('----------------NER----------------')
    #     print(processed_text)
    #     print('----------------NER----------------')
    #     ners = np.array([])
    #     for text in processed_text:
    #         text = self.nlp(text)
    #         ner = np.array([token.ent_type_ for token in text if(token.ent_type_ not in ["X","SPACE"])])
    #         print('----------------NER222222----------------')
    #         print(ner)
    #         print('----------------NER222222----------------')
    #         ners = np.append(ners,ner)

    #     # implement NER using nltk
    #     # ne_tree = ne_chunk(self.POS())
    #     # ner = [i.label() for i in ne_tree if hasattr(i, 'label')]
    #     return ners


# TODO: read from clean dataset folder that has a file called processed_text.txt
file_path = "clean_dataset/first_20_lines.txt"

with open(file_path, "r", encoding="utf8") as file:
    text = [next(file) for _ in range(10)]

# with open("clean_dataset/first_20_lines.txt", "w", encoding="utf8") as file:
#     file.writelines(text)
# make data string instead of list
text = "\n".join(text)
print(text)
"""
#TODO: applying 
1) bag of words 
2) tfidf 
3) word embeddings on this text but output should be a list of each line that contain the features..
therefore, if I have 20 lines ,, should have 20 lists
"""


# read first 100 lines from the text
applied_features = ArabicTextFeatures(text)
# applying_bow = applied_features.tfidf()
# with open("clean_dataset/bow.txt", "w", encoding="utf8") as writing_bow:
#     for line in applying_bow:
#         writing_bow.write(str(line) + "\n")


# cores = multiprocessing.cpu_count()

# w2v_model = Word2Vec(
#     min_count=20,
#     window=2,
#     vector_size=20,
#     sample=6e-5,
#     alpha=0.03,
#     min_alpha=0.0007,
#     negative=20,
#     workers=cores - 1,
# )
file_path = './SG_300_3_400/w2v_SG_300_3_400_10.model'
word_embed = Word2Vec.load(file_path)
word_embeddings = applied_features.word_embeddings(word_embed)
print("Length of word embeddings: ", len(word_embeddings))

# write word embeddings to file
with open("clean_dataset/word_embeddings.txt", "w", encoding="utf8") as writing_embeddings:
    for line in word_embeddings:
        writing_embeddings.write(str(line) + "\n")
        



