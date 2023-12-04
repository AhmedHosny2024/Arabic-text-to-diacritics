import re
import os


class TextProcessor:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder

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
        text = re.sub(r"http\S+", "", text)
        # remove emails
        text = re.sub(r"\S+@\S+", "", text)
        # remove hashtags
        text = re.sub(r"#\S+", "", text)
        # remove mentions
        text = re.sub(r"@\S+", "", text)
        # remove brackets
        text = re.sub(r"\[.*?\]", "", text)
        return text

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

    def process_file(self, input_path, output_path):
        text = self.read_text(input_path)
        print("Before preprocessing data:", len(text))
        clean_text = self.preprocess_text(text)
        print("After preprocessing data:", len(clean_text))
        self.write_to_file(output_path, clean_text)


# processor = TextProcessor("dataset", "clean dataset")
# processor.process_file("train.txt", "clean_train.txt")
