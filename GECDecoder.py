import torch
import torch.nn as nn

from GECAttention import BahdanauAttention


class GECDecoder(nn.Module):
    """
    One-directional LSTM Decoder (left-to-right) with Bahdanau attention.

    It begins from the encoder's final hidden and cell state (a summary of what the
    encoder read), but it is no longer limited to that summary: at every step it also
    attends over ALL encoder outputs to build a context vector. This lets the decoder
    look back at any source character while writing each output character, which fixes
    the information bottleneck of the plain last-hidden-state design.

    In training, teacher forcing is used, meaning the decoder receives the correct
    previous character at each step. It still makes its own predictions (logits), but
    they are not fed back in during training (that happens later, at inference).

    Because the attention context depends on the decoder's own running hidden state,
    the target is processed one character at a time instead of a single
    LSTM call over the whole sequence.

    logit is a raw score showing how confident the decoder is about its prediction.
    Higher score means more confident.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,   # vector size for each character
        hidden_dim,      # size of hidden
        num_layers,      # how many LSTM layers are stacked on top of each other
        pad_index,       # index used for <PAD>
        encoder_output_dim=None,  # size of each encoder output; defaults to 2*hidden_dim (bidirectional encoder)
        attention_dim=None,       # size of the shared attention space; defaults to hidden_dim
    ):
        # nn.Module needs its own setup to run before we add our layers.
        super().__init__()

        # The encoder is bidirectional with hidden size hidden_dim, so each of its
        # outputs is 2*hidden_dim wide
        if encoder_output_dim is None:
            encoder_output_dim = 2 * hidden_dim
        if attention_dim is None:
            attention_dim = hidden_dim
        self.encoder_output_dim = encoder_output_dim

        ### Embedding layer ###
        # Turns each character id into a learnable vector

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_index,  # <PAD> is 0
        )

        ### Attention ###
        # Decides which encoder outputs to focus on at each decoding step
        self.attention = BahdanauAttention(
            encoder_output_dim=encoder_output_dim,
            decoder_hidden_dim=hidden_dim,
            attention_dim=attention_dim,
        )

        ### The LSTM ###
        # This is the actual recurrent network. It walks across the sequence
        # and keeps an internal memory (hidden + cell state) that it updates at
        # every character.
        #
        # Each step now reads the character embedding AND the attention context
        # ("input feeding"), so the input size grows by encoder_output_dim.
        self.lstm = nn.LSTM(
            input_size=embedding_dim + encoder_output_dim,  # embedding of the char + context vector
            hidden_size=hidden_dim,     # size of internal memory
            num_layers=num_layers,      # number of stacked LSTM layers
            batch_first=True,           # tensors are shaped (batch, time, features)
            bidirectional=False,        # decoder is left-to-right
        )

        # Output projection. It sees both the LSTM output and the context vector,
        # so it can use the attended source information directly when guessing the
        # next character.
        self.output_projection = nn.Linear(hidden_dim + encoder_output_dim, vocab_size)

    def forward(self, decoder_inputs, hidden, cell, encoder_outputs, mask):
        """
        Teacher forcing with attention.
        The whole target sequence is given at once. At each step the model always  sees the
        correct previous character, (not its own predictions) plus a context vector.

        """
        # Turning the character ids into vectors
        # (batch, tgt_len) -> (batch, tgt_len, embedding_dim)
        embedded = self.embedding(decoder_inputs)

        # Project the encoder outputs once, reuse at every step
        projected_encoder = self.attention.project_encoder(encoder_outputs)

        tgt_len = embedded.size(1)
        step_outputs = []   # per-step LSTM outputs, stacked after the loop
        step_contexts = []  # per-step attention contexts, stacked after the loop

        # Walk through the target one character at a time, because each step's
        # attention relies on the hidden state produced by the previous step
        for t in range(tgt_len):
            # The query is the current top-layer hidden state. At t=0 this is the encoder's final hidden state
            query = hidden[-1]  # (batch, hidden_dim)

            # Build the context vector
            context, _ = self.attention(query, encoder_outputs, projected_encoder, mask)  # (batch, encoder_output_dim)

            # Feed the character embedding and the context together into the LSTM
            lstm_input = torch.cat([embedded[:, t, :], context], dim=1).unsqueeze(1)

            # One LSTM step, carrying the hidden/cell state forward
            output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
            output = output.squeeze(1)

            # the vocab projection
            step_outputs.append(output)
            step_contexts.append(context)


        outputs  = torch.stack(step_outputs,  dim=1)  # (batch, tgt_len, hidden_dim)
        contexts = torch.stack(step_contexts, dim=1)  # (batch, tgt_len, encoder_output_dim)

        # (batch, tgt_len, hidden_dim + encoder_output_dim) -> (batch, tgt_len, vocab_size)
        logits = self.output_projection(torch.cat([outputs, contexts], dim=2))

        return logits, hidden, cell
