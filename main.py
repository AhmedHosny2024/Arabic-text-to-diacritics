from preprocessing import *
from feature_extraction import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from DataCleaning import DataCleaning
from HMM import HMM
from data_splitting import DataSplitting

def evaluate_features(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    return accuracy





def main():
    input_folder = "dataset"
    output_folder = "clean dataset"
    input_file_name = "train.txt"
    output_file_name = "processed_text.txt"

    processor = DataCleaning(input_folder, output_folder)
    processor.process_file(input_file_name, output_file_name)


    data_splitter = DataSplitting(output_folder,'OutputSplit')
    data_splitter.split_data(output_file_name,'final_out.txt')

    processed_text = processor.read_text(os.path.join('OutputSplit', 'final_out.txt'))
    # print("Processed Text of first 1000:", processed_text[:1000])

    
    feature_extraction = ArabicTextFeatures(processed_text[:80000])
    text_after_feature_extraction = feature_extraction.NER()

    print('---------------text after')
    print(text_after_feature_extraction)
    print('---------------text after')

    n_states = 3 
    n_iter = 100
    tol = 0.01
    verbose = True 
    startprop= np.array([0.5,0.5,0.5])
    transmat = np.array([[0.5,0.5,0.5],[0.5,0.5,0.5],[0.5,0.5,0.5]])
    covar = np.array([[0.5,0.5,0.0],[0.0,0.5,0.5],[0.5,0.0,0.5]])
    HMM_model = HMM(n_states, n_iter, tol, verbose,startprop,transmat,covar)
    HMM_model.fit(text_after_feature_extraction)

    feature_extraction_test = ArabicTextFeatures('ذهب زياد الي المدرسة')
    feature_extraction_test_features = feature_extraction_test.NER()
    print(HMM.predict(feature_extraction_test_features))

    print('----------------finished---------------')

    
    features = ArabicTextFeatures(processed_text)
    bow = features.bag_of_words()
    tfidf = features.tfidf()
    char_ngrams = features.char_ngrams()

    
    # print("Bag of Words:", bow)
    # print("TF-IDF:", tfidf)
    # print("Character N-grams:", char_ngrams)
    
    # test_file_name = "val.txt"
    # test_file_path = os.path.join(input_folder, test_file_name)
    # test_text = processor.read_text(test_file_path)
    
    # bow_accuracy = evaluate_features(bow, test_text)
    # tfidf_accuracy = evaluate_features(tfidf, test_text)

    # print("Bag of Words Accuracy:", bow_accuracy)
    # print("TF-IDF Accuracy:", tfidf_accuracy)
    
    


if __name__ == "__main__":
    main()
