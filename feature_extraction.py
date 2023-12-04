from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class ArabicTextFeatures:
    def __init__(self, text):
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
