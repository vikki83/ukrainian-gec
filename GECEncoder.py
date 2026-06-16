import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GECEncoder(nn.Module):
    """
    Bidirectional LSTM encoder that reads a sentence one character at a time
    in both directions, saves context and hands it to the decoder.

     forward() returns three things:
      - encoder_outputs: the reading at every character (for attention later)
      - hidden, cell: a short summary of the whole sentence, resized for the decoder
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

    def forward(self, encoder_inputs, encoder_lengths):
        '''
        Runs once per batch.

        encoder_inputs  - a batch of sentences as character IDs.
        encoder_lengths - real length of each sentence before padding.
        '''

        # 1) Turning Character IDs into vectors (learnt numbers)
        # Every single character is now a vector instead of one number
        # Shape (batch, max_src_len) -> (batch, max_src_len, embedding_dim)
        embedded = self.embedding(encoder_inputs)
        embedded = self.embedding_dropout(embedded) # prevents overfitting

        # 2) Skipping the padding
        packed = pack_padded_sequence(
            embedded,
            lengths=encoder_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False, # PyTorch sorts sentences on its own
        )

        # 3) Run the BiLSTM, one output vector per character
        # Two outputs:
        #   - packed_outputs - the reading at every character
        #   - final_hidden - the memory after each sentence ends
        # The memory has shape (2 * num_layers, batch, hidden_dim), the rows switch
        # their direction: forward, backward, forward,..... per layer
        packed_outputs, (final_hidden, final_cell) = self.lstm(packed)

        # 4)  Back to a normal tensor (batch, max_src_len, 2 * hidden_dim)
        encoder_outputs, _ = pad_packed_sequence(
            packed_outputs, batch_first=True
        )

        # 5) Two directions into one memory
        # [0::2] - every forward row
        # [1::2] - every backward row.
        # Each is (num_layers, batch, hidden_dim)
        forward_hidden  = final_hidden[0::2]
        backward_hidden = final_hidden[1::2]
        forward_cell    = final_cell[0::2]
        backward_cell   = final_cell[1::2]

        # Two separate memories joined together per layer
        # Now (num_layers, batch, 2 * hidden_dim).
        hidden_concat = torch.cat([forward_hidden, backward_hidden], dim=2)
        cell_concat   = torch.cat([forward_cell,   backward_cell],   dim=2)

        # Changing the size for the Decoder
        # tanh keeps the values within -1 and 1
        hidden = torch.tanh(self.hidden_projection(hidden_concat))
        cell   = torch.tanh(self.cell_projection(cell_concat))

        return encoder_outputs, hidden, cell
