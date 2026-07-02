"""
mBART has its own SentencePiece sub-word tokenizer, so we do not reuse GECVocabulary here. It is like a
translation task, erroneous sentences in Ukrainian are "translated" to correct ones.

"""

import argparse
from datetime import datetime
from pathlib import Path

import torch
from transformers import (
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)


MODEL_NAME = "facebook/mbart-large-50"
LANG       = "uk_UA"
OUTPUT_DIR = "mbart-gec-finetuned"

SRC_TRAIN = "./gec-data/train.src.txt"
TGT_TRAIN = "./gec-data/train.tgt.txt"
SRC_VALID = "./gec-data/valid.src.txt"
TGT_VALID = "./gec-data/valid.tgt.txt"

# Sub-word length caps. The longest UA-GEC sentences are ~700 characters.
# Since Training PC had 16GB V-RAM had to be reduced down to keep V-RAM usage stable.
MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 256

# Memory budget is the whole story here: mBART-large is ~610M parameters, far
# heavier than the LSTM.
PER_DEVICE_BATCH = 4
GRAD_ACCUM_STEPS = 8  # effective batch = 4 * 8 = 32
NUM_EPOCHS       = 3

# "Smoke test": is a quick dry run to check that the whole pipeline works BEFORE launching the real run.
# It trains on just a small slice of the data for a single epoch, so bugs (if any) can be detected soon enough
SMOKE_PAIRS  = 200
SMOKE_EPOCHS = 1
LEARNING_RATE    = 3e-5  # typical fine-tuning learning rate for mBART
WEIGHT_DECAY     = 0.01
WARMUP_RATIO     = 0.1
SEED             = 42



def load_pairs(src_path, tgt_path, drop_markers=True):
    """Read a source/target file pair line-by-line, kept strictly aligned.

    The UA-GEC files contain document IDs (starting with "# " and digits, e.g. "# 0000", "# 0001", ...)
    in the text to mark where each source document begins. They appear identically in both
    the source file and the target file.
    We do not need those ID for training, so we drop them. We read both
    files together and zip line-by-line so the alignment remains the same.
    """

    source_sentences = []
    target_sentences = []

    with open(src_path, encoding="utf-8") as source_file, \
         open(tgt_path, encoding="utf-8") as target_file:

        # zip() walks through both files together, giving us one line from each
        # on every loop.Aligned line-by-line, source_line and target_line always describe the same sentence
        for source_line, target_line in zip(source_file, target_file):

            source_line = source_line.strip()
            target_line = target_line.strip()

            # Skip if empty (a blank line)
            if source_line == "" or target_line == "":
                continue

            # A document marker starts with "# " and the rest is digits.
            # We skip them when drop_markers is True.
            starts_like_marker = source_line.startswith("# ")
            rest_is_a_number = source_line[2:].strip().isdigit()
            if drop_markers and starts_like_marker and rest_is_a_number:
                continue

            # Real sentence pairs
            source_sentences.append(source_line)
            target_sentences.append(target_line)

    return source_sentences, target_sentences


class GECPairsDataset(torch.utils.data.Dataset):
    """Pre-tokenizes every sentence pair once (like GECDataset).

    Each item is the dict the Trainer expects: input_ids / attention_mask for the
    erroneous sentence, and labels for the correct one. Dynamic padding is left to
    the DataCollatorForSeq2Seq at batch time, so nothing is padded here.
    """

    def __init__(self, src, tgt, tokenizer):
        # The modern tokenizer API tokenizes source and target in one call:
        # text_target= routes through the target-language settings so the correct
        # mBART language tokens are attached to each side.
        encoded = tokenizer(
            src,
            text_target=tgt,
            max_length=MAX_SOURCE_LEN,
            truncation=True,
        )
        self.input_ids      = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]
        self.labels         = encoded["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels": self.labels[i],
        }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune mBART50 for Ukrainian GEC.")
    parser.add_argument(
        "--smoke", action="store_true",
        help=f"Quick dry run: only {SMOKE_PAIRS} pairs, {SMOKE_EPOCHS} epoch, to test the pipeline.",
    )
    cli = parser.parse_args()
    num_epochs = SMOKE_EPOCHS if cli.smoke else NUM_EPOCHS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Fine-tuning on: {device}" + ("  [SMOKE TEST]" if cli.smoke else ""))

    tokenizer = MBart50TokenizerFast.from_pretrained(
        MODEL_NAME, src_lang=LANG, tgt_lang=LANG
    )
    model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)

    # Every generated sequence begins with the Ukrainian language token
    model.generation_config.forced_bos_token_id = tokenizer.lang_code_to_id[LANG]
    model.config.forced_bos_token_id = None

    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    train_src, train_tgt = load_pairs(SRC_TRAIN, TGT_TRAIN, drop_markers=True)
    valid_src, valid_tgt = load_pairs(SRC_VALID, TGT_VALID, drop_markers=True)

    # In smoke mode keep only a small slice so the whole loop finishes in minutes.
    if cli.smoke:
        train_src, train_tgt = train_src[:SMOKE_PAIRS], train_tgt[:SMOKE_PAIRS]
        valid_src, valid_tgt = valid_src[:SMOKE_PAIRS], valid_tgt[:SMOKE_PAIRS]

    print(f"Train pairs: {len(train_src)} | Valid pairs: {len(valid_src)}")

    train_dataset = GECPairsDataset(train_src, train_tgt, tokenizer)
    valid_dataset = GECPairsDataset(valid_src, valid_tgt, tokenizer)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100
    )

    # Training arguments
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=num_epochs,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        bf16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,      # keep the lowest-eval-loss checkpoint
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        predict_with_generate=False,      # eval on loss only (generation found to be slow)
        report_to="none",
        seed=SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save the best model + tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved fine-tuned model to {OUTPUT_DIR}/")

    # train vs. eval loss per epoch
    write_training_report(trainer, num_epochs)

    print("Now run: python inference_mbart.py")


def write_training_report(trainer, num_epochs):
    """Summarise train/eval loss per epoch from the trainer's log history."""

    train_by_epoch = {}   # epoch -> list of step train losses
    eval_by_epoch = {}    # epoch -> eval loss
    for entry in trainer.state.log_history:
        epoch = entry.get("epoch")
        if epoch is None:
            continue
        bucket = int(round(epoch))
        if "loss" in entry:
            # If we haven't seen this epoch yet, start an empty list for it,
            # then add this step's training loss to that epoch's list
            if bucket not in train_by_epoch:
                train_by_epoch[bucket] = []
            train_by_epoch[bucket].append(entry["loss"])
        if "eval_loss" in entry:
            eval_by_epoch[bucket] = entry["eval_loss"]

    epochs = sorted(set(train_by_epoch) | set(eval_by_epoch))
    rows = []
    best_epoch, best_eval = None, float("inf")
    for ep in epochs:
        tr = train_by_epoch.get(ep)
        train_loss = sum(tr) / len(tr) if tr else float("nan")
        eval_loss = eval_by_epoch.get(ep, float("nan"))
        if eval_loss == eval_loss and eval_loss < best_eval:   # not NaN
            best_eval, best_epoch = eval_loss, ep
        rows.append((ep, train_loss, eval_loss))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    reports_dir = Path(__file__).resolve().parent / "training reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / f"mbart_training_{timestamp}.txt"
    with open(report_path, "w", encoding="utf-8") as out:
        out.write("mBART-50 Fine-tuning Report\n")
        out.write(f"Date: {timestamp}\n")
        out.write(f"Epochs requested: {num_epochs}\n\n")
        out.write(f"{'epoch':>6}  {'train_loss':>12}  {'eval_loss':>12}\n")
        for ep, train_loss, eval_loss in rows:
            out.write(f"{ep:>6}  {train_loss:>12.4f}  {eval_loss:>12.4f}\n")
        out.write("\n")
        if best_epoch is not None:
            out.write(f"Best (lowest) eval_loss: {best_eval:.4f} at epoch {best_epoch}\n")
            if best_epoch < max(epochs):
                out.write(
                    "Eval loss bottomed out before the final epoch -> later epochs "
                    "added no validation gain (likely starting to overfit). "
                    f"~{best_epoch} epochs is enough for this data/config.\n"
                )
            else:
                out.write(
                    "Eval loss was still dropping at the last epoch -> the model "
                    "may benefit from training longer (try more epochs).\n"
                )
    print(f"Saved training report to: {report_path}")


if __name__ == "__main__":
    main()
