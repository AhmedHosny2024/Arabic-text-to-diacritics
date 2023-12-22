import re
import pyarabic.araby as araby
import os

# TODO: 1) remove all non-Arabic characters , but keep them in array to use them in the future work 
class TextProcessor:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        print("Text Processor Initialized")

    def preprocess_text(self, text):
        # remove non-Arabic characters
        text = re.sub(r"[^\u0600-\u06FF\s]+", "", text)
        # remove extra spaces
        text = re.sub(r"\s+", " ", text)
        # Remove numbers
        text = re.sub(r"\d+", "", text)
        # remove html tags
        text = re.sub(r"<.*?>", "", text)
        # remove urls
        text = re.sub(r"https?\S+", "", text)
        # remove emails
        text = re.sub(r"\S+@\S+", "", text)
        # remove hashtags
        text = re.sub(r"#\S+", "", text)
        # remove mentions
        text = re.sub(r"@\S+", "", text)
        # remove brackets
        text = re.sub(r"\[.*?\]", "", text)
        # remove special characters
        text = re.sub(r"\"|\'|\&\&|\|\||\=\=|\!\=|\<\=|\>\=|\<|\>|\+|\-|\*|\/|\%|\^|\~|\&|\||\!|\=|\<|\>|\(|\)|\{|\}|\[|\]|\:|\;|\?|\_|\`|\@|\#|\$|\\|\,|\.|\“|\”|\«|\»|\٠|\١|\٢|\٣|\٤|\٥|\٦|\٧|\٨|\٩", "", text)
        return text

    def remove_diacritics(self,text):
        return araby.strip_diacritics(text)

    @staticmethod
    def read_text(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    @staticmethod
    def write_to_file(file_path, text):
        output_folder_name = os.path.dirname(file_path)

        if not os.path.exists(output_folder_name):
            os.makedirs(output_folder_name)

        with open(file_path, "w", encoding="utf-8") as file:
            lines = [text[i : i + 200] for i in range(0, len(text), 200)]  # max 200 characters per line
            file.write("\n".join(lines))

    @staticmethod
    def write_to_file_second(dirctory,file_path, text):
        if not os.path.exists(dirctory):
            os.makedirs(dirctory)

        with open(dirctory+"/"+file_path, "w", encoding="utf-8") as file:
            lines = [text[i : i + 200] for i in range(0, len(text), 200)]  # max 200 characters per line
            file.write("\n".join(lines))

    def process_file(self, input_path, output_path):
        text = self.read_text(self.input_folder+"/"+input_path)
        print("Before preprocessing data:", len(text))
        clean_text = self.preprocess_text(text)
        print("After preprocessing data:", len(clean_text))
        without_diacritics = self.remove_diacritics(clean_text)
        self.write_to_file_second(self.output_folder,output_path, clean_text)
        self.write_to_file_second(self.output_folder,'clean_without_diacritics.txt',without_diacritics)
        print("After removing diacritics:", len(without_diacritics))


processor = TextProcessor("Dataset", "CleanDataset")
processor.process_file("train.txt", "clean_train.txt")
