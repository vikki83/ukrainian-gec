import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GECEncoder(nn.Module):
    """
    Bidirectional LSTM Encoder that reads each character at a time
    in both direction and saves context.
    """

    def __init__(
        self,
        vocab_size, # total number of vocabulary characters
        embedding_dim, # size of the vector that represents each character
        hidden_dim, # how many neurons a hidden layer uses
        num_layers, # number of LSTM layers
        pad_index, # integer ID of the <PAD> token
        dropout=0.0, # dropout regularization is off
    ):
        super().__init__()

        ##### the Embedding layer #####
        #
        # Each vocabulary character is represented by indices (e.g. 'a' = 5, 'б' = 6, 'в' = 7, ...)
        # The Embedding Layer turns each character index into a dense vector
        # During training, the model learns those vectors and matches similar characters with similar vectors
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_index, # <PAD> token has a 0 vector, its embedding is never updated during training
        )

        # Dropout helps reduce overfitting by randomly setting some values in the embedding vectors to zero
        # This way the model learns generalisation instead of data memorisation
        # Generalisation is the ability of a trained model to make correct predictions on new, unseen data
        self.embedding_dropout = nn.Dropout(dropout)


        ########  Bidirectional LSTM ############
        #
        # Bidirectional LSTM uses two separate LSTMs, one LSTM reads a sequence left-to-right,
        # and another LSTM reads it right-to-left
        #
        # An LSTM reads the sequence one character at a time.
        # It uses two states:
        #   - hidden state (h): the short-term memory / output at each step
        #   - cell state   (c): the long-term memory (can add or delete information)
        # These states help remember context from earlier characters when reading later ones
        #
        # At every character, outputs of both LSTMs are concatenated,
        # so the output size per step is 2 * hidden_dim

        self.lstm = nn.LSTM(
            input_size=embedding_dim,   # each input character is an embedding vector
            hidden_size=hidden_dim,     # size of each direction's memory
            num_layers=num_layers,      # how many LSTMs to stack on top of each other
            batch_first=True,           # tensors are (batch, time, features), like everywhere else
            bidirectional=True,         # turn on the second, right-to-left reader

            dropout=dropout if num_layers > 1 else 0.0, # to avoid a warning from PyTorch
                                                        # PyTorch only applies LSTM dropout between stacked layers
        )

        ######## Projection - Encoder-Decoder size mismatch ########
        #
        # the Decoder is one-directional and needs a hidden state of only hidden_dim
        # To change the size of the Encoder's final states to fit the size of the Decoder, we use Projection
        #
        # Two separate Projections because the final states shouldn't share weights
        #
        # Linear Layer
        # nn.Linear stores two things:
        #
        # - a weight matrix W of shape (hidden_dim, 2*hidden_dim)
        # - a bias vector b of shape (hidden_dim,)
        # a bias vector is a 1-D vector, with a one element tuple
        #

        self.hidden_projection = nn.Linear(2 * hidden_dim, hidden_dim)
        self.cell_projection   = nn.Linear(2 * hidden_dim, hidden_dim)

