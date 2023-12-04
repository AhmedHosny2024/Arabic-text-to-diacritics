import re
import os


def preprocess_text(text):
    # Remove non-Arabic characters
    text = re.sub(r"[^\u0600-\u06FF\s]+", "", text)

    # Normalize white spaces
    text = re.sub(r"\s+", " ", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # remove html tags
    text = re.sub(r"<.*?>", "", text)

    # remove urls
    text = re.sub(r"http\S+", "", text)

    # remove emails
    text = re.sub(r"\S+@\S+", "", text)

    return text


def read_text(file_path):
    with open(file_path, "r") as file:
        return file.read()


def write_to_file(output_file_path, text):
    # Extract the directory path from the file path
    output_folder_name = os.path.dirname(output_file_path)

    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)

    
    with open(output_file_path, "w", encoding="utf-8") as file:
        lines = [text[i:i+200] for i in range(0, len(text), 120)] # max 200 characters per line
        file.write('\n'.join(lines))
    print(f"Text written to {output_file_path}")


folder_name = "dataset"
output_folder_name = "clean_dataset"

text = read_text(f"{folder_name}/train.txt")[:10000]
clean_text = preprocess_text(text)


output_file_path = f"{output_folder_name}/clean_train.txt"

write_to_file(output_file_path, clean_text)

print(clean_text[::-1])
