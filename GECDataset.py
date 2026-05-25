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

    def collate_fn(self, batch):
        '''
        This method is called by DataLoader to combine many samples
        into one batch. Each batch consists of a 3-tuple (encoder_input, decoder_input, decoder_target)
        coming from __getitem__.
        <PAD> tokens are added to sentences (to fit the longest sentence) so that each batch
         has a tensor of the same length.
        '''
        # PAD index from the saved vocabulary
        pad_index = self.vocab.character_to_index[self.vocab.PAD_TOKEN]

        # Split the batch into three separate lists.
        # The batch before splitting looks like this: [(enc_1, dec_in_1, dec_tgt_1), (enc_2, dec_in_2, dec_tgt_2), ...]

        encoder_inputs = []
        decoder_inputs = []
        decoder_targets = []
        for enc_in, dec_in, dec_tgt in batch:
            encoder_inputs.append(enc_in)
            decoder_inputs.append(dec_in)
            decoder_targets.append(dec_tgt)

        # Stores the length of each sequence
        lengths_list = []
        # This for-loop saves number of tokens of each sequence (before padding) to the list
        for seq in encoder_inputs:
            length = len(seq)
            lengths_list.append(length)
        # Converting the list of integers into a tensor
        encoder_lengths = torch.tensor(lengths_list, dtype=torch.long)

        # pad_sequence makes sure all tensors have the same length as the longest sentence
        encoder_inputs_padded  = pad_sequence(
            encoder_inputs,
            batch_first=True, # The output shape (batch_size, sequence_length)
            padding_value=pad_index # Uses the PAD index from the saved vocabulary
        )
        decoder_inputs_padded  = pad_sequence(
            decoder_inputs,
            batch_first=True,
            padding_value=pad_index
        )
        decoder_targets_padded = pad_sequence(
            decoder_targets,
            batch_first=True,
            padding_value=pad_index
        )

        # Will be used in the training loop
        return (
            encoder_inputs_padded, #  (batch_size, max_src_len) to GECEncoder
            encoder_lengths,  # (batch_size,) for pack_padded_sequence
            decoder_inputs_padded, # (batch_size, max_tgt_len) to GECDecode
            decoder_targets_padded, # (batch_size, max_tgt_len) for loss
        )
