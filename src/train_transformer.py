"""
train_transformer.py - fine-tune XLM-R or MuRIL on SentiMix Hinglish.

    python train_transformer.py                      # final model: MuRIL -> models/muril_final
    python train_transformer.py --model FacebookAI/xlm-roberta-base --out models/xlmr

Needs data/processed/train_final.csv and dev_clean.csv from prepare_data.py.
Use a GPU (Colab / Kaggle free tier). On CPU this will take many hours.
"""
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

p = argparse.ArgumentParser()
p.add_argument("--model", default="google/muril-base-cased")
p.add_argument("--out", default="models/muril_final")
p.add_argument("--epochs", type=int, default=4)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--max_len", type=int, default=128)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--warmup", type=float, default=0.0)   # final run used no warmup
p.add_argument("--train", default="data/processed/train_final.csv")
p.add_argument("--dev", default="data/processed/dev_clean.csv")
args = p.parse_args()
set_seed(args.seed)

LABELS = ["negative", "neutral", "positive"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}


def load(path):
    df = pd.read_csv(path).dropna(subset=["clean_text", "sentiment"]).copy()
    df["clean_text"] = df["clean_text"].astype(str)
    df = df[df["clean_text"].str.strip() != ""]          # drop tweets that were only mentions/URLs
    df["label"] = df["sentiment"].str.lower().str.strip().map(label2id)
    return df.reset_index(drop=True)


train_df, dev_df = load(args.train), load(args.dev)
print(f"train={len(train_df)}  dev={len(dev_df)}  device={'cuda' if torch.cuda.is_available() else 'CPU'}")

tok = AutoTokenizer.from_pretrained(args.model)


class DS(torch.utils.data.Dataset):
    def __init__(self, df):
        self.enc = tok(df["clean_text"].tolist(), truncation=True, max_length=args.max_len)  # dynamic padding
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


train_ds, dev_ds = DS(train_df), DS(dev_df)
model = AutoModelForSequenceClassification.from_pretrained(
    args.model, num_labels=3, label2id=label2id, id2label=id2label)


def metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, -1)
    return {"accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro")}


targs = TrainingArguments(
    output_dir=args.out,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch,
    per_device_eval_batch_size=args.batch * 2,
    learning_rate=args.lr,
    weight_decay=0.01,
    warmup_ratio=args.warmup,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    save_total_limit=1,
    logging_steps=100,
    fp16=torch.cuda.is_available(),
    report_to="none",
    seed=args.seed,
)

trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
                  data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
trainer.train()

out = trainer.predict(dev_ds)
pred = np.argmax(out.predictions, -1)
print("\nDEV RESULTS (best epoch)")
print(classification_report(dev_df["label"], pred, target_names=LABELS, digits=4))
print("Confusion matrix (rows=true, cols=pred; neg/neu/pos):")
print(confusion_matrix(dev_df["label"], pred))

trainer.save_model(args.out)
tok.save_pretrained(args.out)
print("Saved to", args.out)
