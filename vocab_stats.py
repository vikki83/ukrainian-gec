# Check what characters are in the GEC vocabulary


import sys
import unicodedata

from GECVocabulary import src_train, tgt_train

# for Cyrillic letters
sys.stdout.reconfigure(encoding="utf-8")

NUM_SPECIAL_TOKENS = 4


def distinct_characters(source_sentences, target_sentences):
    #Return the set of character types over the union of source and target text.

    chars = set()
    for sentence in source_sentences + target_sentences:
        chars.update(sentence)
    return chars


def classify(chars):

    # Empty lists for character categories
    cyrillic = []
    latin = []
    digits = []
    other = []

    for character in chars:
        # unicodedata.name() gives the official name of a character
        # "" is a fallback name for characters that have none
        name = unicodedata.name(character, "")

        if "CYRILLIC" in name:
            cyrillic.append(character)
        elif "LATIN" in name:
            latin.append(character)
        elif character.isdigit():
            digits.append(character)
        else:
            # Everything else: punctuation, symbols, diacritics, whitespace, emoji
            other.append(character)

    return cyrillic, latin, digits, other


def main():
    chars = distinct_characters(src_train, tgt_train)
    cyrillic, latin, digits, other = classify(chars)

    total_vocab = NUM_SPECIAL_TOKENS + len(chars)

    print(f"Special tokens: {NUM_SPECIAL_TOKENS}")
    print(f"Distinct character types: {len(chars)}")
    print(f"Total vocabulary size: {total_vocab}")
    print("-" * 40)
    print(f"  Cyrillic: {len(cyrillic)}")
    print(f"  Latin: {len(latin)}")
    print(f"  Digits: {len(digits)}")
    print(f"  Other (punct/symbols): {len(other)}")
    print("-" * 40)

    print("Other characters:", "".join(sorted(other)))
    print("Cyrillic:", "".join(sorted(cyrillic)))
    print("Digits:", "".join(sorted(digits)))


if __name__ == "__main__":
    main()
