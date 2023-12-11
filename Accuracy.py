from datasets import load_metric
# ref : actual result 
# pred : predicted result


def BlueScore(ref, pred):
    blue= load_metric("sacrebleu")
    return blue.compute(predictions=pred, references=ref)

# return a dictionary of rouge scores
# reough1 : unigram
# reough2 : bigram
# reoughL : longest common subsequence
# rougeLsum : average of the longest common subsequence
def RougeScore(ref, pred):
    rouge = load_metric("rouge")
    return rouge.compute(predictions=pred, references=ref)

# we need to normalize the text in preprocessing before doing this 
def WER(ref, pred):
    wer = load_metric("wer")
    return wer.compute(predictions=pred, references=ref)

def DER(ref, pred):
    der = load_metric("der")
    return der.compute(predictions=pred, references=ref)

# symbole error rate between two diacritized Arabic text.
def SER(ref, pred):
    ser = load_metric("ser")
    return ser.compute(predictions=pred, references=ref)