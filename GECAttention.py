import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):
    """
    Without attention the encoder reads the whole sentence one character at a time and
    summarizes it in a context vector.

    When the decoder receives such a context vector, it generates less accurate predictions since
    a vector of a fixed length loses some information.

    With attention the decoder sees all encoder outputs plus its final state.

    The additive form:
        score(s, h) = v^T * tanh(W_dec * s + W_enc * h)
    add the decoder state to the encoder output for each source position, squash it with `tanh` (-1 and 1)
    """

    def __init__(self, encoder_output_dim, decoder_hidden_dim, attention_dim):
        super().__init__()

        # W_enc projects each encoder output into a vector space of size attention_dim
        # No bias: the bias would be constant across positions and cancels in the softmax
        self.encoder_projection = nn.Linear(encoder_output_dim, attention_dim, bias=False)

        # W_dec projects the decoder's current hidden state into the same vector space
        self.decoder_projection = nn.Linear(decoder_hidden_dim, attention_dim, bias=False)

        # v outputs one single number (how relevant the source position is)
        self.energy = nn.Linear(attention_dim, 1, bias=False)

    def project_encoder(self, encoder_outputs):
        """
        Project the encoder outputs into vector space.

        The encoder's outputs (readings) remain the same while decoding.
        The decoder computes it once per batch and reuses it at every decoding step.
        """
        return self.encoder_projection(encoder_outputs)

    def forward(self, decoder_hidden, encoder_outputs, projected_encoder, mask):
        """
        It is called at every decoding step.
        """
        # Project the decoder current state
        # unsqueeze(1): broadcast it across every source
        projected_decoder = self.decoder_projection(decoder_hidden).unsqueeze(1)

        # relevance score for every character
        scores = self.energy(torch.tanh(projected_encoder + projected_decoder)).squeeze(2) # the additive form

        # Any paddings are set to negative infinity
        scores = scores.masked_fill(~mask, float("-inf"))

        # Softmax turns the scores into probability-like weights that add up to 1
        attention_weights = torch.softmax(scores, dim=1)

        # Weighted sum of encoder outputs = the context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention_weights
