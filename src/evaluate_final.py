"""
evaluate_final.py - run a trained model ONCE on the SentiMix test set and on the
YouTube comments (out-of-domain check). Saves predictions for error analysis.

    python evaluate_final.py --model models/muril --youtube yt_hinglish_comments_dataset.csv

Only run this after you have finished tuning on the dev set.
"""
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from prepare_data import clean_text

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--test", default="data/processed/test_clean.csv")
p.add_argument("--youtube", default=None)
p.add_argument("--max_len", type=int, default=128)
args = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSequenceClassification.from_pretrained(args.model).to(dev).eval()
LABELS = [model.config.id2label[i] for i in range(3)]


@torch.no_grad()
def predict(texts, bs=64):
    preds = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=args.max_len,
                  padding=True, return_tensors="pt").to(dev)
        preds += model(**enc).logits.argmax(-1).cpu().tolist()
    return [LABELS[i] for i in preds]


def report(name, y_true, y_pred):
    print(f"\n===== {name} ({len(y_true)} examples) =====")
    print(f"accuracy {accuracy_score(y_true, y_pred):.4f} | "
          f"macro-F1 {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(classification_report(y_true, y_pred, labels=LABELS, digits=4))
    print(confusion_matrix(y_true, y_pred, labels=LABELS))


test = pd.read_csv(args.test)
test["clean_text"] = test["clean_text"].fillna("").astype(str)
test["pred"] = predict(test["clean_text"].tolist())
report("SentiMix TEST", test["sentiment"], test["pred"])
test.to_csv("test_predictions.csv", index=False)

if args.youtube:
    yt = pd.read_csv(args.youtube)
    yt["clean_text"] = yt["comment"].astype(str).apply(clean_text)
    yt["sentiment"] = yt["sentiment"].str.lower()
    yt["pred"] = predict(yt["clean_text"].tolist())
    report("YouTube (out-of-domain)", yt["sentiment"], yt["pred"])
    yt.to_csv("youtube_predictions.csv", index=False)
