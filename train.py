import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from GECVocabulary import vocab, src_train, tgt_train, load_sentences, SRC_VALIDATE, TGT_VALIDATE
from GECDataset import GECDataset
from GECEncoder import GECEncoder
from GECDecoder import GECDecoder
from GECSeq2Seq import GECSeq2Seq


# Hyperparameters
NUM_EPOCHS    = 40
BATCH_SIZE    = 64
EMBEDDING_DIM = 64
HIDDEN_DIM    = 256
NUM_LAYERS    = 1
LEARNING_RATE = 1e-3
GRAD_CLIP     = 1.0   # threshold for the gradients
SEED          = 42    # fixed seed so that each run is not random anymore
PATIENCE      = 5     # stops if valid loss is not improving for n-epochs


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


# Data
VOCAB_SIZE = len(vocab.character_to_index)
PAD = vocab.character_to_index[vocab.PAD_TOKEN]


src_valid = load_sentences(SRC_VALIDATE)
tgt_valid = load_sentences(TGT_VALIDATE)

# Builds the train and validation sets as GECDataset objects
train_dataset = GECDataset(src_train, tgt_train, vocab)
valid_dataset = GECDataset(src_valid, tgt_valid, vocab)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=train_dataset.collate_fn,
)
valid_loader = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=valid_dataset.collate_fn,
)


# The Model
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
)
model = GECSeq2Seq(encoder, decoder).to(device)

loss_fn   = nn.CrossEntropyLoss(ignore_index=PAD)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Training
# It runs one epoch of training, updates the weights and returns the average loss for that epoch.
def train_one_epoch():

    model.train()
    total_loss = 0.0
    num_batches = 0

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
    f.write(f"Patience = {PATIENCE}\n\n")
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
