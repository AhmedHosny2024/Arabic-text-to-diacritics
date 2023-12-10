import numpy as np
from hmmlearn import MultinomialHMM

class HMM:
    # n_states: number of hidden states
    # n_iter: number of iterations
    # tol: tolerance
    # verbose: verbose
    # startprop: start probability
    # transmat: transition matrix
    # covar: emission probability
    def __init__(self, n_states, n_iter, tol, verbose,startprop,transmat,covar):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.hmm = None
        self.startprop = startprop
        self.transmat = transmat
        self.covar = covar

    # X is a list of lists of integers (each integer is the index of the word in the vocabulary) 
    def fit(self, X):
        self.hmm = MultinomialHMM(n_components=self.n_states, n_iter=self.n_iter, tol=self.tol, verbose=self.verbose)
        self.hmm.startprob_=self.startprop
        self.hmm.transmat_=self.transmat
        self.hmm.emissionprob_=self.covar
        self.hmm.fit(X)
        return self.hmm
    
    def predict(self, X):
        return self.hmm.predict(X)
    
    