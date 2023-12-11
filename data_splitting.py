import re
import os

class DataSplitting:
    def __init__(self,input_folder, output_folder):
        print("Object created")
        self.input_folder = input_folder
        self.output_folder = output_folder

    def split_arabic_corpus(self,corpus):
        training_data = []
        sentences = []
        segments = corpus.split('.')
        for segment in segments:
            words = segment.split()
            sentences.extend(words)

        training_data = [sentence for sentence in sentences if len(sentence) <=600]

        additional_training_data = []
        for sentence in sentences:
            if len(sentence) >600:
                split_sentences = re.split(r'[،؛;:]', sentence)
                additional_training_data.extend(split_sentences)

        final_training_data = training_data + additional_training_data

        final_training_data = [sentence for sentence in final_training_data if len(sentence) <= 600]

        return " ".join(final_training_data)
        

    @staticmethod
    def write_to_file(dirctory,file_path, text):
        if not os.path.exists(dirctory):
            os.makedirs(dirctory)

        with open(dirctory+"/"+file_path, "w", encoding="utf-8") as file:
            lines = [text[i : i + 200] for i in range(0, len(text), 200)]  # max 200 characters per line
            file.write("\n".join(lines))

    @staticmethod
    def read_text(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def split_data(self,input_path,output_path):
        text = self.read_text(self.input_folder+"/"+input_path)
        print('----------------text----------------')
        print(len(text))
        print('----------------text----------------')
        splitted_text = self.split_arabic_corpus(text)
        self.write_to_file(self.output_folder,output_path, splitted_text)
        return splitted_text



split_object = DataSplitting('Data Cleaning','SplittedOut')
splitted = split_object.split_data('DataCleaning.txt','SplittedOut.txt')
print('----------------splitted----------------')
print(len(splitted))
print('----------------splitted----------------')

