

# Declaring source and target files for training and validation (with errors and corrected)
SRC_TRAIN = "./gec-data/train.src.txt"
TGT_TRAIN = "./gec-data/train.tgt.txt"
SRC_VALIDATE = "./gec-data/valid.tgt.txt"
TGT_VALIDATE = "./gec-data/valid.tgt.txt"

'''
This is the Vocabulary class for building the character-based vocabulary. 
It includes special tokens for LSTM, namely: 
    <PAD> for the same sentence length; its index is 0
    <UNK> for the tokens that were not seen during the training; its index is 1
    <SOS> start of the sentence; its index is 2
    <EOS> end of the sentence; its index is 3
'''
class Vocabulary:

    PAD_TOKEN = '<PAD>'
    SOS_TOKEN = '<SOS>'
    EOS_TOKEN = '<EOS>'
    UNK_TOKEN = '<UNK>'

    def __init__(self):
        self.character_to_index = {
            self.PAD_TOKEN: 0,
            self.SOS_TOKEN: 1,
            self.EOS_TOKEN: 2,
            self.UNK_TOKEN: 3,
        }
        self.index_to_character = {}

''' 
The function load_sentences() reads every single sentence from the files, 
cleans them (by removing \n characters)
and saves them in a list.
'''
def load_sentences(filepath):
    sentences = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            clean_line = line.strip()
            if clean_line:
                sentences.append(clean_line)
    return sentences

# The list with source sentences and the list with target sentences are saved in two separate variables
src_train = load_sentences(SRC_TRAIN)
tgt_train = load_sentences(TGT_TRAIN)
print(src_train)
