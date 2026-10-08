# Hinglish Sentiment Classification

Three-class sentiment classification (negative / neutral / positive) for code-mixed Hindi-English (Hinglish) tweets. The project compares classical TF-IDF baselines with a fine-tuned **MuRIL** transformer.

**Best result: MuRIL, 71.5% test accuracy and 0.72 macro-F1.**

---

## Results

| Model | Features | Dev accuracy |
|---|---|---|
| Logistic Regression | Word TF-IDF | 60.4% |
| Logistic Regression | Character TF-IDF (3-5 grams) | 62.4% |
| **MuRIL (fine-tuned)** | Subword tokens | **65.2%** (dev macro-F1 0.656) |

**Final test-set result (MuRIL, evaluated once on the untouched test set):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Negative | 0.750 | 0.726 | 0.737 |
| Neutral | 0.621 | 0.672 | 0.645 |
| Positive | 0.801 | 0.752 | 0.776 |
| **Accuracy** | | | **0.715** |
| **Macro-F1** | | | **0.720** |

Neutral is the hardest class: most errors are neutral tweets predicted as negative or positive, and the reverse.

---

## Pipeline

1. **Preprocessing** (`prepare_data.py`): cleans the raw text and writes `train_clean.csv`, `dev_clean.csv` and `test_clean.csv`.
2. **Label-conflict removal:** some different tweets become identical after cleaning but carry different labels. In train, 47 such groups (147 rows) exist, and 44 of them are exact ties, so they cannot be resolved by majority vote. These rows are removed, giving `train_final.csv` with 13,853 rows.
3. **MuRIL fine-tuning** on `train_final.csv`, with the best epoch selected on dev macro-F1.
4. **Evaluation** on the test set, used only once at the end.

Dev and test sets are not modified. A few conflicting-label groups remain there (3 in dev, 1 in test).

### MuRIL settings

| Setting | Value |
|---|---|
| Base model | `google/muril-base-cased` |
| Epochs | 4 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Max sequence length | 128 |
| Model selection | Best dev macro-F1 |

---

## Setup and usage

The notebook is written for **Google Colab** (GPU runtime recommended for MuRIL).

```bash
pip install requirements.txt
```

1. Open `v2_cleaned.ipynb` in Colab and switch to a GPU runtime.
2. Zip your project folder (`hinglish-sentiment-analysis/` containing `prepare_data.py` and the raw data) and upload it when the first cell asks.
3. Run all cells from top to bottom.

The notebook saves the fine-tuned model to `models/muril_final`. To classify new text:
