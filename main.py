from preprocessing import *
from feature_extraction import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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

    processor = TextProcessor(input_folder, output_folder)
    processor.process_file(os.path.join(input_folder, input_file_name), os.path.join(output_folder, output_file_name))


    processed_text = processor.read_text(os.path.join(output_folder, output_file_name))
    print("Processed Text:", processed_text[:1000])
    
   

    
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
