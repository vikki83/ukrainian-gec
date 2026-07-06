import os

import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from GECVocabulary import vocab, src_train, tgt_train, load_sentences, SRC_VALIDATE, TGT_VALIDATE
from GECDataset import GECDataset, BucketBatchSampler
from GECEncoder import GECEncoder
from GECDecoder import GECDecoder
from GECSeq2Seq import GECSeq2Seq


# Hyperparameters
NUM_EPOCHS    = 50
BATCH_SIZE    = 48    # Chosen on a 16 GB RTX 4060 Ti after measuring peak VRAM + throughput

EMBEDDING_DIM = 64
HIDDEN_DIM    = 256
NUM_LAYERS    = 1
ATTENTION_DIM = 128   # size of the Bahdanau attention space (smaller = less GPU memory)
MAX_LEN       = 800   # skip train/valid pairs longer than this.
LEARNING_RATE = 1e-3
PROGRESS_EVERY = 25   # print a within-epoch progress line every n batches
GRAD_CLIP     = 1.0   # threshold for the gradients
SEED          = 42    # fixed seed so that each run is not random anymore
PATIENCE      = 5     # stops if valid loss is not improving for n-epochs

# Batch sentences of similar length together (big speedup with the attention
# decoder, which loops over the longest target in each batch). Set to False to
# fall back to the old plain-shuffle behaviour and reproduce earlier runs.
USE_LENGTH_BUCKETING = True
BUCKET_MULTIPLIER    = 50   # megabatch = BATCH_SIZE * this. Larger = tighter length
                            # grouping but less epoch-to-epoch
                            # variety in how sentences are paired into batches.
SAFE_FULL_BATCH_LEN  = 450  # memory guard for the attention decoder. A full BATCH_SIZE
                            # batch is used while sentences are <= this length; beyond it
                            # the batch shrinks (memory ~ B * len^2), so long-sentence
                            # batches no longer overflow VRAM into shared RAM (the cause
                            # of the ~1000s spill batch)


# Reproducibility for the results in the term paper
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# For reproducibility everything needs to be deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")


VOCAB_SIZE = len(vocab.character_to_index)
PAD = vocab.character_to_index[vocab.PAD_TOKEN]


src_valid = load_sentences(SRC_VALIDATE)
tgt_valid = load_sentences(TGT_VALIDATE)

# Builds the train and validation sets as GECDataset objects
# MAX_LEN caps sentence length so the attention decoder fits in GPU memory
# Inference in inference.py still covers all validation sentences, since it does  not use this dataset.
# The ERRANT score is computed on the full valid set
train_dataset = GECDataset(src_train, tgt_train, vocab, max_length=MAX_LEN)
valid_dataset = GECDataset(src_valid, tgt_valid, vocab, max_length=MAX_LEN)

if USE_LENGTH_BUCKETING:
    # Key on the longer of (encoder input, decoder input) so both the encoder
    # padding and the decoder loop length shrink. In GEC src and tgt are nearly
    # the same length, so this is essentially the sentence length.
    train_lengths = [
        max(len(enc), len(dec))
        for enc, dec in zip(train_dataset.encoder_inputs, train_dataset.decoder_inputs)
    ]
    train_batch_sampler = BucketBatchSampler(
        train_lengths, batch_size=BATCH_SIZE,
        bucket_multiplier=BUCKET_MULTIPLIER, seed=SEED,
        max_tokens_sq=BATCH_SIZE * (SAFE_FULL_BATCH_LEN ** 2),
    )
    # With batch_sampler we must NOT also pass batch_size/shuffle/drop_last.
    train_loader = DataLoader(
        train_dataset, batch_sampler=train_batch_sampler,
        collate_fn=train_dataset.collate_fn,
    )
else:
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )

# Validation is left as fixed, unshuffled batches so the reported valid loss
# keeps exactly the same definition across runs (early-stopping signal).
valid_loader = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=valid_dataset.collate_fn,
)


encoder = GECEncoder(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    pad_index=PAD,
)
decoder = GECDecoder(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    pad_index=PAD,
    attention_dim=ATTENTION_DIM,
)
model = GECSeq2Seq(encoder, decoder).to(device)

loss_fn   = nn.CrossEntropyLoss(ignore_index=PAD)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Training
# It runs one epoch of training, updates the weights and returns the average loss for that epoch
def train_one_epoch():

    model.train()
    total_loss = 0.0
    num_batches = 0
    total_batches = len(train_loader)  # how many batches make up one epoch

    # Clock for the within-epoch progress lines. perf_counter() is a steady,
    # high-resolution timer (unaffected by system clock changes). We remember
    # when the previous progress line was printed so the next one can report how
    # many seconds elapsed since it - this shows how fast batches are flowing.
    last_print_time = time.perf_counter()

    for enc_in, enc_lens, dec_in, dec_tgt in train_loader:
        enc_in  = enc_in.to(device)
        dec_in  = dec_in.to(device)
        dec_tgt = dec_tgt.to(device)

        # logits has shape (B, T, V) which stands for batch, time and vocabulary
        logits = model(enc_in, enc_lens, dec_in)  # runs GECSeq2Seq.forward and returns the raw score per character

        # CrossEntropyLoss is not three dimensional, so we need to reshape
        loss = loss_fn(
            logits.reshape(-1, VOCAB_SIZE), # shape (B*T, V)
            dec_tgt.reshape(-1), #  (B*T,)
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1 # Adding a processed batch

        # Within-epoch progress. One epoch is slow (the attention decoder runs
        # one step at a time), so without this the screen looks frozen for
        # minutes and the run seems stuck. flush=True forces the line out right
        # away instead of letting Python buffer it.
        if num_batches % PROGRESS_EVERY == 0 or num_batches == total_batches:
            now = time.perf_counter()
            elapsed = now - last_print_time  # seconds since the previous line
            last_print_time = now
            print(
                f"  epoch {epoch:2d} | batch {num_batches:4d}/{total_batches} "
                f"| running train loss {total_loss / num_batches:.4f} "
                f"| +{elapsed:5.1f}s",
                flush=True,
            )

    return total_loss / num_batches


# Validation
@torch.no_grad()  # disables gradient tracking so it is faster
def validate():
    model.eval()
    total_loss = 0.0
    num_batches = 0
    total_correct = 0  # correctly predicted characters
    total_chars   = 0  # total chars in target

    for enc_in, enc_lens, dec_in, dec_tgt in valid_loader:
        enc_in  = enc_in.to(device)
        dec_in  = dec_in.to(device)
        dec_tgt = dec_tgt.to(device)

        logits = model(enc_in, enc_lens, dec_in)
        loss = loss_fn(
            logits.reshape(-1, VOCAB_SIZE),
            dec_tgt.reshape(-1),
        )
        total_loss += loss.item()
        num_batches += 1

        # Char-level accuracy
        preds = logits.argmax(dim=-1)            # (B, T)
        mask  = (dec_tgt != PAD)                 # ignore PAD positions
        total_correct += ((preds == dec_tgt) & mask).sum().item()
        total_chars   += mask.sum().item()

    return total_loss / num_batches, total_correct / total_chars


# Saving a report after each training
REPORT_DIR = "classification reports"
os.makedirs(REPORT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
report_path = os.path.join(REPORT_DIR, f"report_{timestamp}.txt")

with open(report_path, "w", encoding="utf-8") as f:
    f.write("Training Report\n")
    f.write(f"Date: {timestamp}\n\n")
    f.write(f"Number of epochs = {NUM_EPOCHS}\n")
    f.write(f"Batch size = {BATCH_SIZE}\n")
    f.write(f"Embedding dimension = {EMBEDDING_DIM}\n")
    f.write(f"Hidden dimension = {HIDDEN_DIM}\n")
    f.write(f"Number of layers= {NUM_LAYERS}\n")
    f.write(f"Learning rate= {LEARNING_RATE}\n")
    f.write(f"Patience = {PATIENCE}\n")
    f.write(f"Length bucketing = {USE_LENGTH_BUCKETING} (bucket multiplier {BUCKET_MULTIPLIER}, safe full-batch len {SAFE_FULL_BATCH_LEN})\n\n")
    f.write("Epoch | Train loss | Valid loss | Valid character accuracy\n")


### Early stopping ###
#
# To avoid overfitting, we track the lowest validation loss seen so far. If the model spots
# that val loss is not getting lower than the best so far, PATIENCE counter is increasing. If it does get lower, the counter resets
# and the weights are remembered as the new "best model".
# If the counter reaches PATIENCE, validation is not improving anymore, so we stop early and
# keep the best weights instead of the last ones
best_valid_loss = float("inf")  # no loss yet, infinity as default
best_epoch      = 0
epochs_no_improve = 0
best_state = None # a saved copy of the best weights


# Main loop
for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = train_one_epoch()
    valid_loss, valid_acc = validate()
    line = f"{epoch:5d} | {train_loss:10.4f} | {valid_loss:10.4f} | {valid_acc:14.4f}"
    print(line)

    with open(report_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    # Epoch improved?
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        best_epoch = epoch
        epochs_no_improve = 0
        # Keep a copy of the current weights, because later epochs will
        # keep changing the model in memory
        # model.state_dict() is a dictionary: layer name -> its weight tensor.
        # We copy it entry by entry into best_state
        best_state = {}
        for layer_name, weight_tensor in model.state_dict().items():
            # .detach() cut the copy off from training, so it won't change later
            # .cpu() moves it from GPU to ordinary RAM
            # .clone()  makes a real copy (not just a reference)
            best_state[layer_name] = weight_tensor.detach().cpu().clone()
    else:
        epochs_no_improve += 1
        # Stop if validation hasn't improved for PATIENCE epochs in a row.
        if epochs_no_improve >= PATIENCE:
            stop_msg = (
                f"Early stopping at epoch {epoch}: "
                f"no improvement for {PATIENCE} epochs "
                f"(best was epoch {best_epoch}, valid loss {best_valid_loss:.4f})."
            )
            print(stop_msg)
            with open(report_path, "a", encoding="utf-8") as f:
                f.write("\n" + stop_msg + "\n")
            break

# Keep best epoch
summary = f"Best model: epoch {best_epoch} (valid loss {best_valid_loss:.4f})."
print(summary)
with open(report_path, "a", encoding="utf-8") as f:
    f.write(summary + "\n")
print(f"Saved training report to {report_path}")


# Load the best weights back into the model
if best_state is not None:
    model.load_state_dict(best_state)
torch.save(model.state_dict(), "gec_seq2seq.pt") # state_dict keeps the learned weights
print("Saved model to gec_seq2seq.pt")
