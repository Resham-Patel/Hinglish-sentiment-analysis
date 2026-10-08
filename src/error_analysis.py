"""
error_analysis.py - where does the final MuRIL model go wrong on the SentiMix test set?

Run after `evaluate_final.py` (it creates test_predictions.csv):
    python error_analysis.py
"""
import re
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("test_predictions.csv")
df["clean_text"] = df["clean_text"].fillna("").astype(str)
df["correct"] = df["sentiment"] == df["pred"]
df["words"] = df["clean_text"].str.split().str.len()
df["has_emoji"] = df["text"].astype(str).str.contains(r"[\U0001F000-\U0001FAFF\u2600-\u27BF]")


def show(title, groupby):
    print(f"\n--- {title} ---")
    rows = []
    for name, g in df.groupby(groupby, observed=True):
        rows.append({"group": name, "n": len(g),
                     "accuracy": round(accuracy_score(g["sentiment"], g["pred"]), 3),
                     "macro_f1": round(f1_score(g["sentiment"], g["pred"], average="macro"), 3)})
    print(pd.DataFrame(rows).to_string(index=False))


print(f"Overall: accuracy {df.correct.mean():.4f}, macro-F1 "
      f"{f1_score(df.sentiment, df.pred, average='macro'):.4f}, "
      f"{(~df.correct).sum()} errors out of {len(df)}")

# 1. Language mix: share of Hindi tokens among Hindi+English tokens (from the CONLL tags)
df["language_mix"] = pd.cut(df["hin_ratio"], [-0.01, 0.25, 0.75, 1.0],
                            labels=["mostly English (<25% Hindi)", "mixed (25-75%)", "mostly Hindi (>75%)"])
show("By language mix", "language_mix")

# 2. Length
df["length"] = pd.cut(df["words"], [-1, 5, 12, 25, 1000], labels=["0-5 words", "6-12", "13-25", "26+"])
show("By tweet length (after cleaning)", "length")

# 3. Emoji
show("Has emoji", "has_emoji")

# 4. Which mistakes are most common
print("\n--- Most common mistakes (true -> predicted) ---")
err = df[~df.correct]
print(err.groupby(["sentiment", "pred"]).size().sort_values(ascending=False).to_string())

# 5. Examples of each mistake type
print("\n--- Example mistakes ---")
for (true, pred), g in err.groupby(["sentiment", "pred"]):
    print(f"\n[true={true}, predicted={pred}]  ({len(g)} tweets)")
    for t in g["clean_text"].sample(min(5, len(g)), random_state=1):
        print("  -", re.sub(r"\s+", " ", str(t))[:200])

err.to_csv("test_errors.csv", index=False)
print("\nSaved all errors to test_errors.csv")
