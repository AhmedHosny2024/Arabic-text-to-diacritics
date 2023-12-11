# Read the files
with open('./dataset/val.txt', 'r') as file:
    val_content = set(file.readlines())

with open('./dataset/train.txt', 'r') as file:
    train_content = set(file.readlines())


# merge the two sets
merged_content = val_content.union(train_content)


# merge the three files in real folder
with open('./realset/test_git.txt', 'r') as file:
    file1_content = set(file.readlines())

with open('./realset/train_git.txt', 'r') as file:
    file2_content = set(file.readlines())

with open('./realset/val_git.txt', 'r') as file:
    file3_content = set(file.readlines())

# merge the three sets
merged_content_real = file1_content.union(file2_content, file3_content)


# get sentences that are in the real folder but not in the dataset folder
diff = merged_content_real.difference(merged_content)

# write the sentences in a file
with open('./dataset/diff.txt', 'w') as file:
    file.writelines(diff)
