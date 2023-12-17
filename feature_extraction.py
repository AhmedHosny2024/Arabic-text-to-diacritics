from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import spacy
import numpy as np
from nltk.tokenize import word_tokenize
# from nltk import pos_tag,ne_chunk
class ArabicTextFeatures:
    def __init__(self, text):
        # load model for e3rab each word and extract meaning of each word
        self.nlp = spacy.load("xx_sent_ud_sm")
        self.doc=self.nlp(text)
        self.text = text
        print("Arabic Text Features Initialized")

    def bag_of_words(self):
        vectorizer = CountVectorizer()
        bow = vectorizer.fit_transform([self.text]) 
        return bow.toarray()

    def tfidf(self):
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([self.text]) 
        return tfidf.toarray()

    def char_ngrams(self, n=2):
        ngrams = [self.text[i : i + n] for i in range(len(self.text) - n + 1)] 
        return ngrams
    
    # part of speech tagging
    def POS(self):
        # apply pos_ method on each token in the text return token.pos_ for each token except X(unknown word), SPACE(whitespace)
        pos = [token.pos_ for token in self.doc if(token.pos_ not in ["X","SPACE"])]
        
        # implement POS using nltk
        # tokenized_text = word_tokenize(self.text)
        # pos = pos_tag(tokenized_text, lang='ar')
        return pos
    
    # named entity recognition
    def NER(self):
        # apply ent_type_ method on each token in the text return token.ent_type_ for each token except X(unknown word), SPACE(whitespace)


        print('----------------NER0000000----------------')
        print(self.text)
        print('----------------NER0000000----------------')
        processed_text = self.text.split('\n')
        print('----------------NER----------------')
        print(processed_text)
        print('----------------NER----------------')
        ners = np.array([])
        for text in processed_text:
            text = self.nlp(text)
            ner = np.array([token.ent_type_ for token in text if(token.ent_type_ not in ["X","SPACE"])])
            print('----------------NER222222----------------')
            print(ner)
            print('----------------NER222222----------------')
            ners = np.append(ners,ner)
        
        # implement NER using nltk
        # ne_tree = ne_chunk(self.POS())
        # ner = [i.label() for i in ne_tree if hasattr(i, 'label')]
        return ners