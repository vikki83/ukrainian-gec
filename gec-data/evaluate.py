#!/usr/bin/env python3
"""Evaluate the model output.

Usage:
    evaluate.py <corrected> [--no-tokenize] --m2 <path_to_m2>
    evaluate.py (-h | --help)

Options:
    -h --help           Show this screen.
    --no-tokenize       Do not tokenize the submission
    --layer <layer>     Annotation layer to evaluate: `gec-only` or `gec-fluency`.

<corrected> is the path to the model output. If --no-tokenize is not specified,
the input will be tokenized before evaluation.

"""

import argparse
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# On Windows the errant_parallel/errant_compare subprocesses open the Ukrainian
# files with the cp1252 codepage and crash. Forcing UTF-8 here makes the spawned
# child processes inherit it, so the script works even when run without
# PYTHONUTF8=1 set in the environment (e.g. the IDE Run button).
os.environ["PYTHONUTF8"] = "1"

import spacy
import stanza


def tokenize(text: str) -> [str]:
    if not hasattr(tokenize, "nlp"):
        tokenize.nlp = stanza.Pipeline(lang="uk", processors="tokenize")
    nlp = tokenize.nlp

    tokenized = " ".join([t.text for t in nlp(text).iter_tokens()])
    return tokenized


def tokenize_file(input_file: Path, output_file: Path):
    with open(input_file, encoding="utf-8") as f, open(output_file, "w", encoding="utf-8") as out:
        for line in f:
            line = line.rstrip("\n")
            tokenized = tokenize(line)
            out.write(tokenized + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the model output.")
    # Both paths default so the script can be run bare (e.g. the IDE Run button)
    # against the standard validation outputs without passing any arguments.
    # Anchor the defaults to the script's own location (gec-data/ and the repo
    # root above it) so they resolve regardless of the current working directory.
    script_dir = Path(__file__).resolve().parent      # .../gec-data
    repo_root = script_dir.parent                     # repo root
    parser.add_argument("corrected", type=str, nargs="?", default=str(repo_root / "valid.hyp.txt"),
            help="Path to the model output (default: <repo>/valid.hyp.txt)")
    parser.add_argument("--m2", type=str, default=str(script_dir / "valid.m2"),
            help="Path to the golden annotated data (.m2 file) (default: gec-data/valid.m2)")
    parser.add_argument("--no-tokenize", action="store_true", help="Do not tokenize the submission")
    args = parser.parse_args()
    tmp = Path(tempfile.gettempdir())

    # Make sure we have spacy resources downlaoded
    # spaCy 3 (required by ERRANT 3.x) dropped the "en" shortcut; use the model name.
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spacy resources...", file=sys.stderr)
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)

    # Tokenize corrected file if needed
    if args.no_tokenize:
        tokenized_path = args.corrected
    else:
        print("Tokenizing submission...", file=sys.stderr)
        tokenized_path = tmp / f"unlp.target.tok"
        tokenize_file(args.corrected, tokenized_path)
    print(f"Tokenized: {tokenized_path}", file=sys.stderr)

    # Get the source text out of m2
    source_path = tmp / f"unlp.source.tok"
    with open(args.m2, encoding="utf-8") as f, open(source_path, "w", encoding="utf-8") as out:
        for line in f:
            if line.startswith("S "):
                out.write(line[2:])

    # Align tokenized submission with the original text with Errant
    m2_target = tmp / "unlp.target.m2"
    subprocess.run(["errant_parallel", "-orig", source_path, "-cor", tokenized_path, "-out", m2_target], check=True)
    print(f"Aligned submission: {m2_target}", file=sys.stderr)

    # Evaluate. Capture the errant_compare output instead of letting it write
    # straight to the terminal, so we can both print it AND save it to a report.
    summary = subprocess.run(
        ["errant_compare", "-hyp", m2_target, "-ref", args.m2],
        capture_output=True, text=True, encoding="utf-8").stdout
    detailed = subprocess.run(
        ["errant_compare", "-hyp", m2_target, "-ref", args.m2, "-ds"],
        capture_output=True, text=True, encoding="utf-8").stdout

    # Still show the results on screen, exactly as before.
    print(summary)
    print(detailed)

    # Save a timestamped report next to the training reports so every eval
    # (including future models) leaves a permanent, comparable record.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    reports_dir = repo_root / "classification reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / f"errant_{timestamp}.txt"
    with open(report_path, "w", encoding="utf-8") as out:
        out.write("ERRANT Evaluation Report\n")
        out.write(f"Date: {timestamp}\n\n")
        out.write(f"Hypotheses: {args.corrected}\n")
        out.write(f"Gold (m2):  {args.m2}\n")
        out.write(f"Tokenized:  {'no (--no-tokenize)' if args.no_tokenize else 'yes (stanza uk)'}\n\n")
        out.write(summary)
        out.write(detailed)
    print(f"Saved report to: {report_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
