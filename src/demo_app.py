"""
demo_app.py - Gradio demo: type a Hinglish sentence, get its sentiment.
Uses the fine-tuned MuRIL model (the final model of this project).

    pip install gradio
    python demo_app.py --model models/muril_final
In Colab a public link is printed (share=True).
"""
import argparse
import numpy as np
import torch
import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from prepare_data import clean_text

LABELS = ["negative", "neutral", "positive"]
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/muril_final")
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"


def load(path):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
    order = [model.config.label2id[l] for l in LABELS]
    return tok, model, order


tok, model, order = load(args.model)


@torch.no_grad()
def predict(text):
    text = (text or "").strip()
    if not text:
        return {}
    cleaned = clean_text(text) or text.lower()
    enc = tok(cleaned, truncation=True, max_length=128, return_tensors="pt").to(device)
    probs = torch.softmax(model(**enc).logits, -1).cpu().numpy()[0][order]
    return {l: float(p) for l, p in zip(LABELS, probs)}


examples = [
    "Movie bahut boring thi yaar, waste of money",
    "Phone ki battery backup ekdum mast hai, camera bhi badhiya",
    "Delivery late thi but product quality achi hai",
    "Kal meeting 5 baje hai office mein",
    "Sarkar ne naya budget pesh kiya aaj",
]

gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, label="Hinglish text", placeholder="Type a Hindi-English sentence..."),
    outputs=gr.Label(num_top_classes=3, label="Predicted sentiment"),
    examples=examples,
    title="Hinglish Sentiment Analysis",
    description="MuRIL fine-tuned on SemEval-2020 SentiMix (Hindi-English tweets). "
                "Trained on tweets, so it works best on social-media style text.",
).launch(share=True)
