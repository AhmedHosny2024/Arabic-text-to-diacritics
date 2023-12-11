import re
import os
import pickle as pkl
from bs4 import BeautifulSoup

# TODO: 1) remove all non-Arabic characters , but keep them in array to use them in the future work 
class DataCleaning:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
    
    def preprocess_text(self, text):
        # Data Cleaning: All characters except the Arabic letters, diacritics, and punctuation are removed from the corpus, 
        # which is essential as it keeps only the characters that contribute to the diacritization to be learned by the models. 
        # In inference, we designed algorithms that can restore the removed characters and put them back in their right places.
        # All characters except the Arabic letters, diacritics, and punctuation
        with open('./pickles/ARABIC_LETTERS_LIST.pickle', 'rb') as file:
            ARABIC_LETTERS_LIST = pkl.load(file)
        with open('./pickles/DIACRITICS_LIST.pickle', 'rb') as file:
            DIACRITICS_LIST = pkl.load(file)
        # Remove HTML tags using BeutifulSoup
        text = BeautifulSoup(text, "html.parser").get_text()
        # Remove URLs 
        text = re.sub(r"http[s|S]\S+", "", text,flags=re.MULTILINE)
        text = re.sub(r"www\S+", "", text,flags=re.MULTILINE)
        # Remove English letters 
        text = re.sub(r"[A-Za-z]+", "", text,flags=re.MULTILINE)
        # Remove Kashida Arabic character
        text = re.sub(r"\u0640", "", text,flags=re.MULTILINE)
        # Add space before and after the numbers
        text = re.sub(r"(\d+)", r" \1 ", text,flags=re.MULTILINE)
        # Remove multiple whitespaces
        text = re.sub(r"\s+", " ", text,flags=re.MULTILINE)
        # removes SHIFT+J Arabic character
        text = re.sub(r"\u0691", "", text,flags=re.MULTILINE)
        # remove english numbers
        text = re.sub(r"[0-9]+", "", text,flags=re.MULTILINE)
        # remove arabic numbers
        text = re.sub(r"[٠-٩]+", "", text,flags=re.MULTILINE)
        # taken from the original repo code to fixes diacritics positions and remove unneeded or misplaced ones 
        def fix_diacritics(content):
            content = re.sub(r'اً', 'ًا', content)
            # add diacritics to the end of the word
            content = re.sub(r'(?P<char>[' + ''.join(ARABIC_LETTERS_LIST) + DIACRITICS_LIST[-1] + '])\s+(?P<diac>[' + ''.join(DIACRITICS_LIST) + ']+)(?P<brek>[\s+]|\Z)', r'\g<char>\g<diac>\g<brek>', content)
            # remove diacritics from the end of the word 
            content = re.sub(r'(?P<char>[^' + ''.join(ARABIC_LETTERS_LIST) + ''.join(DIACRITICS_LIST) + '])[' + ''.join(DIACRITICS_LIST) + ']+', r'\g<char>', content)
            # remove multiple diacritics 
            content = re.sub(r'[' + DIACRITICS_LIST[-1] + ']+', DIACRITICS_LIST[-1], content)
            # remove diacritics from the beginning of the word
            content = re.sub(r'(?P<diac>[' + ''.join(DIACRITICS_LIST[:-1]) + '])[' + ''.join(DIACRITICS_LIST) + ']+', r'\g<diac>', content)
            return content
        # Remove diacritics
        for diacritic in DIACRITICS_LIST:
            text = text.replace(diacritic, "")
        text = fix_diacritics(text)
        return text

    @staticmethod
    def read_text(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    @staticmethod
    def write_to_file(dirctory,file_path, text):
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
        self.write_to_file(self.output_folder,output_path, clean_text)

processor = DataCleaning("Dataset", "Data Cleaning")
processor.process_file("train.txt", "DataCleaning.txt")
