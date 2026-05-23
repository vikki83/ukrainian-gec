# Declaring source and target files (containing error sentences and corrected ones) for training and validation
SRC_TRAIN = "./gec-data/train.src.txt"
TGT_TRAIN = "./gec-data/train.tgt.txt"
SRC_VALIDATE = "./gec-data/valid.src.txt"
TGT_VALIDATE = "./gec-data/valid.tgt.txt"

def load_sentences(filepath):
    '''
    This function reads every single sentence from the source and target files,
    cleans them (by removing \n characters)
    and saves them in a list.
    '''

    sentences = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            clean_line = line.strip()
            if clean_line:
                sentences.append(clean_line)
    return sentences

class GECVocabulary:
    '''
     This is the Vocabulary class for building the character-based vocabulary.
     It includes special tokens for LSTM, namely:
        <PAD> for the same sentence length;
        <SOS> indicates start of the sentence;
        <EOS> signals end of the sentence;
        <UNK> for the tokens that were not seen during the training;
    '''

    PAD_TOKEN = '<PAD>'
    SOS_TOKEN = '<SOS>'
    EOS_TOKEN = '<EOS>'
    UNK_TOKEN = '<UNK>'

    def __init__(self):
        # This dictionary stores key-value pairs,
        # where key is a special token and value is an index ( e.g. '<PAD>' : 0 )
        # It is used during encoding
        self.character_to_index = {
            self.PAD_TOKEN: 0,
            self.SOS_TOKEN: 1,
            self.EOS_TOKEN: 2,
            self.UNK_TOKEN: 3,
        }
        # Reversed key-value pairs from the "character_to_index" dictionary ( e.g. 0: '<PAD>' )
        # It is used during decoding, since the decoder must receive an index first
        self.index_to_character = {}
        for k, v in self.character_to_index.items():
            self.index_to_character[v] = k

    def build_vocabulary(self, source_sentences, target_sentences):

        ua_characters = set() # Set() keeps the key-value pairs in a fixed order (character and its corresponding index)

        # This for-loop adds one character at a time to the list ua_characters
        for sentence in source_sentences + target_sentences:
            for character in sentence:
                ua_characters.add(character)

        # This for-loop checks for every letter from ua_characters if it is included to the character_to_index list
        # and assigns to it the next index in a row, building a new key-value pair.
        # If the letter is already in the list, no duplicates are created
        for character in sorted(ua_characters):
            if character not in self.character_to_index:
                index = len(self.character_to_index)
                self.character_to_index[character] = index
                self.index_to_character[index] = character

        print(f"Tokens in total: {len(self.character_to_index)}")

    def encode(self, sentence, add_sos = False, add_eos = True):
        '''
        This method converts a string (a sentence) into a list of integers.
        There are 2 boolean parameters:

        add_sos needs to be True for the decoder,
        so it knows when to start generating a sentence;

        add_eos needs to be True for both the encoder and the decoder,
        It signals for the encoder as an input end,
        and for the decoder as the stopping condition for sentence generation.
        '''

        indices = [] # An empty list to store the indices
        if add_sos:
            indices.append(self.character_to_index[self.SOS_TOKEN])

        # Checks each character at a time and adds UNK token if it was not seen during vocabulary building
        for character in sentence:
            index = self.character_to_index.get(character, self.character_to_index[self.UNK_TOKEN])
            indices.append(index)

        # Adds the EOS token to the end of the sentence
        if add_eos:
            indices.append(self.character_to_index[self.EOS_TOKEN])

        return indices

    def decode(self, indices):
         '''
         This method converts a list of integers into a string (sentence).
         It is reversed of encode().
         The special tokens are no longer needed (the sentences must be text only).
         '''

         special_token_indices = set(range(4)) # The four tokens with indices from 0 to 3 are the special tokens
         decoded_characters = [] # An empty list to store the characters
         for index in indices:
             if index in special_token_indices:
                 continue # Skips the special tokens
             decoded_characters.append(self.index_to_character[index])
         return "".join(decoded_characters) # Returns full sentences by joining separate letters into words



# The lists with source and target sentences are saved in two separate variables
src_train = load_sentences(SRC_TRAIN)
tgt_train = load_sentences(TGT_TRAIN)

vocab = GECVocabulary()
vocab.build_vocabulary(src_train, tgt_train)

# --------------------------------------------------------------------------------------
# Testing if encode() and decode() work
test_sentence = "Вітаю з днем народження."
encoder_input  = vocab.encode(test_sentence, add_sos=False, add_eos=True)
decoder_input  = vocab.encode(test_sentence, add_sos=True,  add_eos=False)
decoder_target = vocab.encode(test_sentence, add_sos=False, add_eos=True)
decoded = vocab.decode(encoder_input)

print("Encoder input indices:", encoder_input)
print("Decoder input indices:", decoder_input)
print("Decoder target indices:", decoder_target)
print("Sentence decoded?", test_sentence == decoded)
