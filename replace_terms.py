#!/usr/bin/env python3
"""Replace Star Wars terms in row_texts/ with invented equivalents.

The substitution dictionary is the single source of truth in terms_map.json
(original -> fictional). This script loads it and, for every *.txt in
row_texts/, applies a single-pass, longest-match-first, word-boundary-aware
substitution, then writes the result to knowledge_base/.

Original files are never modified. To change a mapping, edit terms_map.json.
"""
import json
import re
from pathlib import Path

BASE = Path("/home/admin/architecture-rag")
SRC = BASE / "row_texts"
DST = BASE / "knowledge_base"
MAP_PATH = BASE / "terms_map.json"

_ARTICLE_RE = re.compile(r"\b([Aa]n?)\s+([A-Za-z])")


def fix_articles(text):
    """Repair a/an agreement disturbed by substitution (vowel-letter heuristic)."""
    def repl(m):
        art, first = m.group(1), m.group(2)
        new = "an" if first.lower() in "aeiou" else "a"
        if art[0].isupper():
            new = new.capitalize()
        return f"{new} {first}"
    return _ARTICLE_RE.sub(repl, text)


def build_pattern(terms):
    # Longest first so multi-word / longer keys win in the ordered alternation.
    keys = sorted(terms, key=len, reverse=True)
    alt = "|".join(re.escape(k) for k in keys)
    # Word-boundary via lookarounds (handles keys ending in digits/letters).
    # Case-sensitive: separates proper nouns (Force, Empire) from common words.
    return re.compile(r"(?<![\w])(?:" + alt + r")(?![\w])")


def main():
    terms = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    pattern = build_pattern(terms)
    DST.mkdir(exist_ok=True)

    def sub(m):
        return terms[m.group(0)]

    total = 0
    files = sorted(SRC.glob("*.txt"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    for path in files:
        text = path.read_text(encoding="utf-8")
        new_text, n = pattern.subn(sub, text)
        new_text = fix_articles(new_text)
        (DST / path.name).write_text(new_text, encoding="utf-8")
        total += n
        print(f"{path.name}: {n} replacements")

    print(f"\nTerms loaded from {MAP_PATH.name}: {len(terms)}")
    print(f"Total replacements: {total}")
    print(f"Documents written: {len(files)} -> {DST}")


if __name__ == "__main__":
    main()
