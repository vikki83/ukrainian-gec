"""Error analysis

To run for mBART50: python error_analysis.py valid_mbart.hyp.txt --m2 gec-data/valid.m2

`evaluate.py` shows that the model fixes only about 26% of all errors. This script breaks that number
apart by the gold error category (Spelling, Punctuation, G/Case, F/Calque, ...) so you can see what the model
misses. It also prints a few real example sentences for each category.

The plan, step by step:
    1. Tokenize the model's output the same way evaluate.py does.
    2. Use `errant_parallel` to turn that output into an .m2 file (a list of
       edits), just like evaluate.py does.
    3. Read BOTH the gold .m2 and the model's .m2 into Python.
    4. For every gold error, check if the model made the same edit.
         - same place in the sentence and same correction  -> "caught"
         - otherwise "missed"
    5. Count caught/missed for each category, save a few missed examples.
    6. Print and save a report.

"""

import argparse
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Forcing UTF-8 for Ukrainian
os.environ["PYTHONUTF8"] = "1"
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# The tokenizer function is from evaluate.py, the model output is tokenized in the exact same
# way it was when it was scored
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "gec-data"))
from evaluate import tokenize_file

MAX_EXAMPLES = 8

MODEL_OUTPUT = str(REPO_ROOT / "valid.hyp.txt")
GOLD_M2 = str(REPO_ROOT / "gec-data" / "valid.m2")


def read_m2(path):
    """
    Read an .m2 file into a simple Python structure.

    We return a list of sentences. Each sentence is a dictionary:
        {
            "tokens": ["тут", "іде", ...],     # the source words
            "edits_by_annotator": {            # one edit list per annotator
                "0": [ (start, end, category, correction), ... ],
                "1": [ ... ],
            },
        }
    """
    sentences = []
    tokens = None
    edits_by_annotator = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("S "):
                # A new sentence begins. Save the source tokens.
                tokens = line[2:].split(" ")
                edits_by_annotator = {}

            elif line.startswith("A "):
                parts = line[2:].split("|||")
                start, end = parts[0].split(" ")
                start, end = int(start), int(end)
                category = parts[1]
                correction = parts[2]
                annotator = parts[5]

                if annotator not in edits_by_annotator:
                    edits_by_annotator[annotator] = []

                # "noop" means "no error here", so we do not store it as an edit
                if category != "noop" and start != -1:
                    edit = (start, end, category, correction)
                    edits_by_annotator[annotator].append(edit)

            elif line == "":
                # A blank line ends the current sentence
                if tokens is not None:
                    sentences.append({
                        "tokens": tokens,
                        "edits_by_annotator": edits_by_annotator,
                    })
                tokens = None
                edits_by_annotator = {}

        # The file might not end with a blank line, so we save the last sentence
        if tokens is not None:
            sentences.append({
                "tokens": tokens,
                "edits_by_annotator": edits_by_annotator,
            })

    return sentences


def edit_fingerprint(edit):
    """
    Turn an edit into a comparable "fingerprint".

    Two edits count as the same correction if they cover the same token span and replace it with the same
    text. We tidy the correction's spaces so that, for example, "столі" and " столі " are treated as equal.

    """
    start, end, category, correction = edit
    clean_correction = " ".join(correction.split())
    return (start, end, clean_correction)


def build_model_m2(corrected_path, gold_m2_path, no_tokenize):
    """
    Run the same alignment steps evaluate.py uses, and return the model .m2.

    This tokenizes the model output and feeds it to `errant_parallel`, which
    figures out what edits turn the source sentence into the model's output.
    """
    tmp_dir = Path(tempfile.gettempdir())

    # Step 1: tokenize the model output (unless it is already done)
    if no_tokenize:
        tokenized_path = corrected_path
    else:
        print("Tokenizing the model output (Stanza uk)...", file=sys.stderr)
        tokenized_path = tmp_dir / "ea.model.tok"
        tokenize_file(corrected_path, tokenized_path)

    # Step 2: pull the source sentences out of the gold .m2 (the "S" lines)
    # We must use the same source as the gold file, so the token positions match
    source_path = tmp_dir / "ea.source.tok"
    with open(gold_m2_path, encoding="utf-8") as f, open(source_path, "w", encoding="utf-8") as out:
        for line in f:
            if line.startswith("S "):
                out.write(line[2:])

    # Step 3: errant_parallel compares source vs. model output and writes edits
    model_m2 = tmp_dir / "ea.model.m2"
    subprocess.run(
        ["errant_parallel", "-orig", source_path, "-cor", tokenized_path, "-out", model_m2],
        check=True,
    )
    return model_m2


def pick_best_annotator(edits_by_annotator, model_fingerprints):
    """
    Choose one of the two human annotators to compare against.

    The two annotators sometimes correct a sentence very differently. ERRANT
    compares against whichever annotator is most favourable to the model, so we
    do the same.
    """
    best_edits = []
    best_caught = -1
    best_size = None

    for edits in edits_by_annotator.values():
        # Count how many of this annotator's edits the model also made
        caught = 0
        for edit in edits:
            if edit_fingerprint(edit) in model_fingerprints:
                caught += 1

        is_better = caught > best_caught
        is_tie_but_smaller = caught == best_caught and (best_size is None or len(edits) < best_size)
        if is_better or is_tie_but_smaller:
            best_edits = edits
            best_caught = caught
            best_size = len(edits)

    return best_edits


def main():
    # --- Command-line arguments (default to the paths set at the top) --------
    parser = argparse.ArgumentParser(description="Analyse which error types the model misses.")
    parser.add_argument("corrected", nargs="?", default=MODEL_OUTPUT,
            help="Path to the model output (default: valid.hyp.txt)")
    parser.add_argument("--m2", default=GOLD_M2,
            help="Path to the gold .m2 file (default: gec-data/valid.m2)")
    parser.add_argument("--no-tokenize", action="store_true", help="Do not tokenize the model output")
    args = parser.parse_args()

    # Get both files as Python lists of sentences
    model_m2_path = build_model_m2(args.corrected, args.m2, args.no_tokenize)
    gold_sentences = read_m2(args.m2)
    model_sentences = read_m2(model_m2_path)

    if len(gold_sentences) != len(model_sentences):
        sys.exit(f"Different number of sentences: gold={len(gold_sentences)} "
                 f"model={len(model_sentences)}. Did inference.py finish?")

    # Counters: how many errors of each category, and how many were caught -
    total_per_category = {}    # category -> number of gold errors
    caught_per_category = {}   # category -> number the model also fixed
    examples_per_category = {}  # category -> list of (sentence, original, correction)

    # Go sentence by sentence
    for gold, model in zip(gold_sentences, model_sentences):
        # The model .m2 has a single annotator. Get its edits (or an empty list).
        model_edits = []
        for edits in model["edits_by_annotator"].values():
            model_edits = edits
            break
        model_fingerprints = set(edit_fingerprint(e) for e in model_edits)

        # Pick the human annotator to compare against for this sentence
        gold_edits = pick_best_annotator(gold["edits_by_annotator"], model_fingerprints)

        # Check each gold edit, if the model made the same correction
        for edit in gold_edits:
            start, end, category, correction = edit

            total_per_category[category] = total_per_category.get(category, 0) + 1
            if category not in caught_per_category:
                caught_per_category[category] = 0
            if category not in examples_per_category:
                examples_per_category[category] = []

            if edit_fingerprint(edit) in model_fingerprints:
                caught_per_category[category] += 1
            else:
                # The model missed this error. Save it as an example (up to a cap)
                if len(examples_per_category[category]) < MAX_EXAMPLES:
                    sentence_text = " ".join(gold["tokens"])
                    original = " ".join(gold["tokens"][start:end])
                    if original == "":
                        original = "(nothing — a word should be inserted)"
                    shown_correction = correction if correction != "" else "(delete)"
                    examples_per_category[category].append(
                        (sentence_text, original, shown_correction))

    # Report text
    total_errors = sum(total_per_category.values())
    total_caught = sum(caught_per_category.values())
    overall_recall = total_caught / total_errors if total_errors > 0 else 0.0

    report_lines = []
    report_lines.append("Error Analysis Report: caught vs. missed by error type")
    report_lines.append("Date: " + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    report_lines.append("Model output: " + os.path.basename(args.corrected))
    report_lines.append("Gold file: " + os.path.basename(args.m2))
    report_lines.append("")
    report_lines.append(f"Total gold errors: {total_errors}")
    report_lines.append(f"Caught by model: {total_caught}")
    report_lines.append(f"Overall recall: {overall_recall:.3f}")
    report_lines.append("")

    # Sort the categories so the most common errors come first
    categories_sorted = sorted(total_per_category, key=lambda c: total_per_category[c], reverse=True)

    # Table
    report_lines.append(f"{'Category':<24}{'Total':>7}{'Caught':>8}{'Missed':>8}{'Recall':>9}")
    report_lines.append("-" * 56)
    for category in categories_sorted:
        total = total_per_category[category]
        caught = caught_per_category[category]
        missed = total - caught
        recall = caught / total if total > 0 else 0.0
        report_lines.append(f"{category:<24}{total:>7}{caught:>8}{missed:>8}{recall:>9.3f}")

    # The example sentences, grouped by category
    report_lines.append("")
    report_lines.append("=" * 56)
    report_lines.append("EXAMPLES OF MISSED ERRORS")
    report_lines.append("=" * 56)
    for category in categories_sorted:
        examples = examples_per_category[category]
        if len(examples) == 0:
            continue
        missed = total_per_category[category] - caught_per_category[category]
        report_lines.append("")
        report_lines.append(f"--- {category}  (showing {len(examples)} of {missed} missed) ---")
        for i, (sentence_text, original, correction) in enumerate(examples, 1):
            report_lines.append(f"[{i}] sentence: {sentence_text}")
            report_lines.append(f"    should fix: \"{original}\"  ->  \"{correction}\"")

    report = "\n".join(report_lines) + "\n"

    print(report)
    reports_dir = REPO_ROOT / "classification reports"
    reports_dir.mkdir(exist_ok=True)
    out_path = reports_dir / ("error_analysis_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt")
    with open(out_path, "w", encoding="utf-8") as out:
        out.write(report)
    print("Saved report to: " + str(out_path), file=sys.stderr)


if __name__ == "__main__":
    main()
