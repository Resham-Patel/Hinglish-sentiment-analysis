"""
prepare_data.py  -  CONLL -> clean CSV for train / dev / test (SentiMix Hinglish)

Keeps uid, text, sentiment, per-token language tags and a few language-mix
features. Handles Windows (CRLF) and Unix line endings.

Also writes train_final.csv: train_clean.csv with every group of tweets that
share the same clean_text but have DIFFERENT labels removed (47 groups, 147 rows
-> 13,853 training rows). This is the file the final MuRIL model is trained on.

Usage (put raw files in data/raw/):
    python prepare_data.py
Outputs data/processed/{train,dev,test}_clean.csv and train_final.csv
"""
import csv
import re
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/processed")

# ---------- cleaning (improved version of your preprocess.py) ----------
URL_SPLIT = re.compile(r"https?\s*:?\s*//\s*t\s*\.?\s*co\s*/?\s*\S*", re.I)
URL_NORM = re.compile(r"https?://\S+|www\.\S+", re.I)
# consecutive mentions incl. split usernames like "@ Shehla _ Rashid" or "@ user"
MENTION = re.compile(r"@\s*\w+(?:\s*_\s*\w+)*")


def clean_text(text: str) -> str:
    text = text.lower()
    text = URL_SPLIT.sub(" ", text)
    text = URL_NORM.sub(" ", text)
    text = MENTION.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------- mojibake repair ----------
# The official test file is double-encoded (UTF-8 bytes read as cp1252), so
# emojis show up as "ðŸ˜…" and "…" as "â€¦". Train/dev are fine; we only repair
# tokens that show the tell-tale pattern.
MOJIBAKE = re.compile(r"[ðâÃÂ][\x80-\xff\u0152\u0153\u0160\u0161\u0178\u017d\u017e\u0192\u02c6\u02dc\u2013-\u203a\u20ac\u2122]")


def fix_mojibake(tok: str) -> str:
    if not MOJIBAKE.search(tok):
        return tok
    try:
        return tok.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    try:  # bytes that cp1252 leaves undefined (0x81, 0x8d, 0x8f, 0x90, 0x9d)
        raw = bytes(ord(c) if ord(c) < 256 else c.encode("cp1252")[0] for c in tok)
        return raw.decode("utf-8")
    except Exception:
        return tok


# ---------- CONLL parsing ----------
def parse_conll(path):
    """Yield dicts: uid, sentiment (or None), tokens, langs."""
    text = Path(path).read_text(encoding="utf-8").replace("\r\n", "\n").strip()
    for block in re.split(r"\n\s*\n", text):
        lines = [l for l in block.split("\n") if l.strip()]
        if not lines or not lines[0].startswith("meta"):
            continue
        meta = lines[0].split("\t")
        uid = meta[1].strip()
        sentiment = meta[2].strip().lower() if len(meta) > 2 else None
        tokens, langs = [], []
        for line in lines[1:]:
            parts = line.split("\t")
            tok = fix_mojibake(parts[0].strip())
            if not tok:
                continue
            tokens.append(tok)
            langs.append(parts[1].strip() if len(parts) > 1 else "O")
        yield {"uid": uid, "sentiment": sentiment, "tokens": tokens, "langs": langs}


def load_gold_labels(path):
    labels = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            labels[row["Uid"].strip()] = row["Sentiment"].strip().lower()
    return labels


def build_csv(conll_path, out_path, gold=None):
    rows = []
    for ex in parse_conll(conll_path):
        sentiment = ex["sentiment"] or (gold or {}).get(ex["uid"])
        if sentiment is None:
            raise ValueError(f"No label for uid {ex['uid']}")
        text = " ".join(ex["tokens"])
        lang = [l.lower() for l in ex["langs"]]
        n_word = sum(1 for l in lang if l in ("hin", "eng"))
        rows.append({
            "uid": ex["uid"],
            "text": text,
            "clean_text": clean_text(text),
            "sentiment": sentiment,
            "lang_tags": " ".join(ex["langs"]),
            "hin_ratio": round(lang.count("hin") / n_word, 3) if n_word else 0.0,
        })
    OUT.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    dist = {}
    for r in rows:
        dist[r["sentiment"]] = dist.get(r["sentiment"], 0) + 1
    print(f"{out_path}: {len(rows)} rows  {dist}")


def remove_label_conflicts(src, dst):
    """Drop all rows whose clean_text appears with more than one sentiment label."""
    with open(src, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    labels = {}
    for r in rows:
        labels.setdefault(r["clean_text"], set()).add(r["sentiment"])
    conflicting = {t for t, ls in labels.items() if len(ls) > 1}
    kept = [r for r in rows if r["clean_text"] not in conflicting]
    with open(dst, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(kept)
    print(f"{dst}: removed {len(conflicting)} conflicting groups "
          f"({len(rows) - len(kept)} rows) -> {len(kept)} rows")


if __name__ == "__main__":
    build_csv(RAW / "train_14k_split_conll.txt", OUT / "train_clean.csv")
    build_csv(RAW / "dev_3k_split_conll.txt", OUT / "dev_clean.csv")
    gold = load_gold_labels(RAW / "test_labels_hinglish.txt")
    build_csv(RAW / "Hindi_test_unalbelled_conll_updated.txt",
              OUT / "test_clean.csv", gold=gold)
    remove_label_conflicts(OUT / "train_clean.csv", OUT / "train_final.csv")
