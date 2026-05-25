import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from GECVocabulary import vocab, src_train, tgt_train, load_sentences, SRC_VALIDATE, TGT_VALIDATE


# Hyperparameters to be adjusted
NUM_EPOCHS    = 5
BATCH_SIZE    = 64
EMBEDDING_DIM = 64
HIDDEN_DIM    = 256
NUM_LAYERS    = 1
LEARNING_RATE = 1e-3
GRAD_CLIP     = 1.0   # threshold for the gradients


# Default device GPU; if not available switch to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")


VOCAB_SIZE = len(vocab.character_to_index)
PAD = vocab.character_to_index[vocab.PAD_TOKEN]

src_valid = load_sentences(SRC_VALIDATE)
tgt_valid = load_sentences(TGT_VALIDATE)

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
    f.write(f"Learning rate= {LEARNING_RATE}\n\n")
    f.write("Epoch | Train loss | Valid loss | Valid character accuracy\n")





for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = train_one_epoch()
    valid_loss, valid_acc = validate()
    line = f"{epoch:5d} | {train_loss:10.4f} | {valid_loss:10.4f} | {valid_acc:14.4f}"
    print(line)
    # Append per epoch so partial runs still leave a usable report.
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

print(f"Saved training report to {report_path}")


# Saving the model
torch.save(model.state_dict(), "gec_seq2seq.pt") # state_dict keeps the learned weights
print("Saved model to gec_seq2seq.pt")
