"""
Mirrors inference.py for the LSTM: it reads the validation source sentences,
produces one corrected sentence per line, and writes them to a hypothesis file
that the SAME evaluation script scores against valid.m2.

"""

import torch
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration


MODEL_DIR   = "mbart-gec-finetuned"
LANG        = "uk_UA"
SRC_VALID   = "./gec-data/valid.src.txt"
OUTPUT_FILE = "valid_mbart.hyp.txt"   # one corrected sentence per line

BATCH_SIZE  = 8
NUM_BEAMS   = 5       # beam search gives better corrections
MAX_GEN_LEN = 400

def is_marker(line):
    return line.startswith("# ") and line[2:].strip().isdigit()


def read_all_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    # Load the fine-tuned model and its tokenizer from disk. Setting ukrainian language to the tokenizer
    tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_DIR, src_lang=LANG, tgt_lang=LANG)
    model = MBartForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
    # Training saved use_cache=False (needed for gradient checkpointing); re-enable
    # the KV cache here so beam search generation runs at full speed
    model.config.use_cache = True
    model.eval()
    print(f"Loaded {MODEL_DIR}/")

    # Read every validation source line into a list
    sentences = read_all_lines(SRC_VALID)
    print(f"Generating corrections for {len(sentences)} lines...")

    # Setting the first generated token Ukrainian
    forced_bos = tokenizer.lang_code_to_id[LANG]

    outputs = []
    # Work through the file in fixed-size batches so we don't run out of memory
    for start in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[start:start + BATCH_SIZE]

        # Marker lines are passed through with no change; only real sentences are sent through the model.
        # "positions" remembers where each real sentence is in the batch

        to_correct, positions = [], []
        for i, line in enumerate(batch):
            if not is_marker(line) and line:
                to_correct.append(line)
                positions.append(i)

        # Start with a copy of the batch
        corrected = list(batch)
        if to_correct:
            # Tokenize the batch of sentences into padded tensors of token ids
            enc = tokenizer(
                to_correct,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_GEN_LEN,
            ).to(device)

            # Generate corrections. no_grad() skips gradient tracking (less memory)
            # since we're only doing inference, not training.
            with torch.no_grad():
                generated = model.generate(
                    **enc,
                    forced_bos_token_id=forced_bos,   # Ukrainian output
                    num_beams=NUM_BEAMS,              # beam search for quality
                    max_length=MAX_GEN_LEN,
                )
            # Turn the generated token ids back into strings, dropping special tokens
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            # Write each correction back into its original position, collapsing
            # any runs of whitespace into single spaces.
            for pos, text in zip(positions, decoded):
                corrected[pos] = " ".join(text.split())

        outputs.extend(corrected)

        # Print a progress update every 100 lines.
        done = min(start + BATCH_SIZE, len(sentences))
        if done % 100 < BATCH_SIZE:
            print(f"  {done}/{len(sentences)}")

    # Write one corrected sentence per line to the file.
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for line in outputs:
            f.write(line + "\n")

    print(f"Wrote corrections to {OUTPUT_FILE}")
    # Remind the user how to score these corrections against the gold M2 file.
    print(f"Now run: python gec-data/evaluate.py {OUTPUT_FILE} --m2 gec-data/valid.m2")


if __name__ == "__main__":
    main()
