import torch
import torch.nn as nn


class GECDecoder(nn.Module):
    """
    One-directional LSTM Decoder (left-to-right) that begins from the encoder's final hidden
    and cell state (summary of what the encoder read).

    It reads this summary, target sentences and uses teacher forcing to produce the correct output.

    In training, teacher forcing is used, meaning the decoder receives correct characters at a time.
    It still makes its own predictions though (logits), but they are not used in training (later for inference ).

    logit is a raw score showing how confident the decoder is about its prediction. Higher score means more confident.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,   # vector size for each character
        hidden_dim,      # size of hidden
        num_layers,      # how many LSTM layers are stacked on top of each other
        pad_index,       # index used for <PAD>
    ):
        # nn.Module needs its own setup to run before we add our layers.
        super().__init__()

        ### Embedding layer ###
        # Turns each character id into a learnable vector

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_index,  # <PAD> is 0
        )

        ### The LSTM ###
        # This is the actual recurrent network. It walks across the sequence
        # and keeps an internal memory (hidden + cell state) that it updates at
        # every character.
        self.lstm = nn.LSTM(
            input_size=embedding_dim,   # it receives one embedding vector per step
            hidden_size=hidden_dim,     # size of internal memory
            num_layers=num_layers,      # number of stacked LSTM layers
            batch_first=True,           # tensors are shaped (batch, time, features)
            bidirectional=False,        # decoder is left-to-right
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, decoder_inputs, hidden, cell):
        """
        Teacher forcing.
        The whole target sequence is fed at once. The model always sees the correct previous character,
        (not its own predictions).
        """
        # Turning the character ids into vectors
        # (batch, max_tgt_len) → (batch, max_tgt_len, embedding_dim)
        embedded = self.embedding(decoder_inputs)

        # Run the LSTM over the whole sequence,
        # `outputs` hold one hidden vector per time step
        # outputs: (batch, max_tgt_len, hidden_dim)
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        # Project every step's hidden vector to vocab-sized scores, so each
        #    position gets a guess for the next character.
        #    (batch, max_tgt_len, hidden_dim) → (batch, max_tgt_len, vocab_size)
        logits = self.output_projection(outputs)

        return logits, hidden, cell
