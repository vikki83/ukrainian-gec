import torch
import torch.nn as nn


class GECSeq2Seq(nn.Module):
    """
    The full sequence-to-sequence model: it holds the encoder and the decoder together and connects them.

    Without it, the encoder and the decoder cannot be a complete model. This class joins them like this:

        source sentence -> encoder -> summary -> decoder -> corrected text

    Because this class wraps both, the training loop can do everything with
    a single model(...) call per batch. It does not call the encoder and
    decoder separately.
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        # Encoder and decoder as submodules
        # model.parameters() automatically includes their parameters
        self.encoder = encoder
        self.decoder = decoder

    @staticmethod
    def _make_mask(encoder_lengths, max_len, device):
        """
        Build the source padding mask the attention needs.

        True where there is a real character, False where the position is <PAD>.
        encoder_lengths counts src + <EOS>, which is exactly the number of valid
        time steps in encoder_outputs.
        """
        lengths = encoder_lengths.to(device)
        positions = torch.arange(max_len, device=device)
        return positions.unsqueeze(0) < lengths.unsqueeze(1)

    def forward(self, encoder_inputs, encoder_lengths, decoder_inputs):
        """
        Teacher-forcing forward pass. Returns next-char logits for every
        position of the target sequence.
        """
        # Encode the source. With attention we keep encoder_outputs (one vector per source character)
        encoder_outputs, hidden, cell = self.encoder(encoder_inputs, encoder_lengths)

        # Mask <PAD> positions
        mask = self._make_mask(encoder_lengths, encoder_outputs.size(1), encoder_outputs.device)

        # Decode with encoder's final state + attention
        logits, _, _ = self.decoder(decoder_inputs, hidden, cell, encoder_outputs, mask)

        return logits

    @torch.no_grad()  # inference only, no gradients needed
    def generate(self, source_sentence, vocab, device=None, max_length=None):
        """
        Greedy decoding: given ONE source sentence, produce a corrected version, one char at a time.

        It is different from forward() since there is no target sequence at inference; the decoder uses its own guesses
        by feeding them back in as the next decoder input.
        """
        self.eval()

        if device is None:
            device = next(self.parameters()).device
        SOS = vocab.character_to_index[vocab.SOS_TOKEN]
        EOS = vocab.character_to_index[vocab.EOS_TOKEN]

        # Encode the source (turning raw input sentence into the numeric tensor)
        # 1-sample batch shape: (1, src_len + 1), where +1 is <EOS>
        # shape = size of each dimension
        src_ids = vocab.encode(source_sentence, add_sos=False, add_eos=True)
        encoder_inputs  = torch.tensor([src_ids], dtype=torch.long, device=device)
        encoder_lengths = torch.tensor([len(src_ids)], dtype=torch.long) # real length

        encoder_outputs, hidden, cell = self.encoder(encoder_inputs, encoder_lengths)

        # There is no padding because we decode one sentence at a time. But the decoder still expects a mask argument
        mask = self._make_mask(encoder_lengths, encoder_outputs.size(1), encoder_outputs.device)

        # Preventing an infinite loop if the sentence never reaches <EOS>
        if max_length is None:
            max_length = len(src_ids) * 2

        next_input = torch.tensor([[SOS]], dtype=torch.long, device=device) # starts with <SOS>
        output_ids = []
        for _ in range(max_length):
            # logits (a score vector, shows how likely it is for each character to come next)
            # After the decoder runs one step, logits has shape (1, 1, vocab_size):
            #   where 1 is batch (one sentence) and one character position,
            #   vocab_size is one raw score for every character in vocabulary
            logits, hidden, cell = self.decoder(next_input, hidden, cell, encoder_outputs, mask)
            next_id = logits.argmax(dim=-1)  # greedy decoding - takes the single best character at each step

            # Stop as soon as EOS comes
            if next_id.item() == EOS:
                break

            output_ids.append(next_id.item())
            # Feed the prediction back in as the next decoder input
            next_input = next_id

        # vocab.decode skips special tokens to have a clean text
        return vocab.decode(output_ids)
