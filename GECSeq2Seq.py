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

    def forward(self, encoder_inputs, encoder_lengths, decoder_inputs):
        """
        Teacher-forcing forward pass. Returns next-char logits for every
        position of the target sequence.
        """
        # Encode the source. We only need the final hidden/cell here
        _, hidden, cell = self.encoder(encoder_inputs, encoder_lengths)

        # 2) Decode using the encoder's final state
        logits, _, _ = self.decoder(decoder_inputs, hidden, cell)

        return logits

    @torch.no_grad()  # inference only, no gradients needed
    def generate(self, source_sentence, vocab, device=None, max_length=None):
        """
        Greedy decoding: given ONE source sentence, produce a corrected version, one char at a time.

        It is different from forward() since there is no target sequence at inference;
         we have to feed each prediction back in as the next decoder input.
        """
        self.eval()

        if device is None:
            device = next(self.parameters()).device
        SOS = vocab.character_to_index[vocab.SOS_TOKEN]
        EOS = vocab.character_to_index[vocab.EOS_TOKEN]

        # Encode the source
        # 1-sample batch shape: (1, src_len + 1) with EOS appended
        src_ids = vocab.encode(source_sentence, add_sos=False, add_eos=True)
        encoder_inputs  = torch.tensor([src_ids], dtype=torch.long, device=device)
        encoder_lengths = torch.tensor([len(src_ids)], dtype=torch.long)

        _, hidden, cell = self.encoder(encoder_inputs, encoder_lengths)

        # Decoder loop, one step at a time
        if max_length is None:
            max_length = len(src_ids) * 2

        next_input = torch.tensor([[SOS]], dtype=torch.long, device=device)
        output_ids = []
        for _ in range(max_length):
            # logits shape: (1, 1, vocab_size) — one step, one batch element.
            logits, hidden, cell = self.decoder(next_input, hidden, cell)
            next_id = logits.argmax(dim=-1)  # (1, 1)

            # Stop as soon as the model emits EOS
            if next_id.item() == EOS:
                break

            output_ids.append(next_id.item())
            # Feed the prediction back in as the next decoder input
            next_input = next_id

        # vocab.decode skips PAD/SOS/EOS/UNK, so the result is clean text
        return vocab.decode(output_ids)
