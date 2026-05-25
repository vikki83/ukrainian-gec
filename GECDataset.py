import torch
from torch.utils.data import Dataset, DataLoader



class GECDataset(Dataset):
    '''
    This PyTorch Dataset stores sentence pairs and returns the encoded source
    and target sentences as tensors, needed for the encoder-decoder model.
    The model receives 2 inputs ("encoder_input", "decoder_input") and
    predicts "decoder_target".
    '''

    def __init__(self, source_sentences, target_sentences, vocabulary):
        '''
        This method coverts all sentence pairs to tensors.

        There are three different lists (for encoder-decoder inputs and decoder
        predictions) that store those sentence pairs.
        Each sentence pair has the same index for all three lists.
        '''

        self.vocab = vocabulary

        # SOS and EOS have fixed indices, so we fetch them here, outside the loop below,
        # to avoid constant dictionary lookups
        sos_index = vocabulary.character_to_index[vocabulary.SOS_TOKEN]
        eos_index = vocabulary.character_to_index[vocabulary.EOS_TOKEN]

        # Sentence pairs will be saved here as tensors (one tensor per pair to three different lists)
        self.encoder_inputs = [] # src + EOS (what the encoder reads)
        self.decoder_inputs = [] # SOS + tgt (what the decoder reads)
        self.decoder_targets = [] # tgt + EOS (what the decoder predicts)

        # zip() matches the source-target pairs that have the same index
        for src_sentence, tgt_sentence in zip(source_sentences, target_sentences):
            '''
            This loop iterates over all sentence pairs and appends them as tensors 
            to the three lists above. 
            '''
            # vocabulary.encode() converts a string to a list with integers
            src_ids = vocabulary.encode(src_sentence, add_sos=False, add_eos=False) # no special tokens needed here
            tgt_ids = vocabulary.encode(tgt_sentence, add_sos=False, add_eos=False)

            # Tensors are created here
            self.encoder_inputs.append(
                torch.tensor(src_ids + [eos_index], dtype=torch.long) # data stored is 64-bit integers
            )
            self.decoder_inputs.append(
                torch.tensor([sos_index] + tgt_ids, dtype=torch.long)
            )
            self.decoder_targets.append(
                torch.tensor(tgt_ids + [eos_index], dtype=torch.long)
            )

    def __len__(self):
        return len(self.encoder_inputs) # Number of sentence pairs

    def __getitem__(self, index):

        # Returning sentence pairs as tensors at n-position
        return (
            self.encoder_inputs[index],
            self.decoder_inputs[index],
            self.decoder_targets[index],
        )

