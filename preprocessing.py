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

    def read_text(self, file_name):
        with open(f"{self.input_folder}/{file_name}", "r", encoding="utf-8") as file:
            return file.read()

    def write_to_file(self, file_name, text):
        output_file_path = f"{self.output_folder}/{file_name}"
        output_folder_name = os.path.dirname(output_file_path)

        if not os.path.exists(output_folder_name):
            os.makedirs(output_folder_name)

        with open(output_file_path, "w", encoding="utf-8") as file:
            lines = [
                text[i : i + 200] for i in range(0, len(text), 200)
            ]  # max 200 characters per line
            file.write("\n".join(lines))
        # print(f"Text written to {output_file_path}")

    def process_file(self, input_file_name, output_file_name):
        text = self.read_text(input_file_name)
        print("Before preprocessing data:", len(text))
        clean_text = self.preprocess_text(text)
        print("After preprocessing data:", len(clean_text))
        self.write_to_file(output_file_name, clean_text)


# Usage
processor = TextProcessor("dataset", "clean dataset")
processor.process_file("train.txt", "clean_train.txt")
