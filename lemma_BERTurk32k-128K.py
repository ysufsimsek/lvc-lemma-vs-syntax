import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    AutoConfig
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================
# CSV DOSYALARI
# =========================
csv_files = [
    "sentences_with_lemmas_and_labels.csv",
    "sentences_with_lemmas_and_labels_lightveb.csv"
]

def read_csv_files(paths):
    texts, labels = [], []

    for path in paths:
        df = pd.read_csv(path)

        texts.extend(df["sentence"].astype(str).tolist())
        labels.extend(df["label"].astype(int).tolist())

    return texts, labels

all_texts, all_labels = read_csv_files(csv_files)

print(f"Toplam cümle sayısı: {len(all_texts)}")

train_texts, test_texts, train_labels, test_labels = train_test_split(
    all_texts,
    all_labels,
    test_size=0.2,
    stratify=all_labels,
    random_state=42
)

train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
test_ds  = Dataset.from_dict({"text": test_texts,  "label": test_labels})

MAX_LEN = 64

def tokenize_fn(batch, tokenizer):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
    )

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

def train_bert_robust(model_name, train_dataset, test_dataset):
    print(f"\n{'='*10} Model Eğitiliyor: {model_name} {'='*10}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_tok = train_dataset.map(lambda e: tokenize_fn(e, tokenizer), batched=True)
    test_tok  = test_dataset.map(lambda e: tokenize_fn(e, tokenizer), batched=True)

    train_tok = train_tok.rename_column("label", "labels")
    test_tok  = test_tok.rename_column("label", "labels")

    train_tok.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_tok.set_format("torch",  columns=["input_ids", "attention_mask", "labels"])

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=2,
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config
    ).to(device)

    args = TrainingArguments(
        output_dir=f"results_{model_name.replace('/', '_')}",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        weight_decay=0.01,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=test_tok,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()

    print(f"\n--- {model_name} Değerlendirme Sonuçları ---")
    eval_results = trainer.evaluate()
    print(eval_results)
    return eval_results

bert32  = "dbmdz/bert-base-turkish-cased"
bert128 = "dbmdz/bert-base-turkish-128k-cased"

res_32  = train_bert_robust(bert32, train_ds, test_ds)
res_128 = train_bert_robust(bert128, train_ds, test_ds)

print(
    f"BERTurk 32K  -> Accuracy: {res_32['eval_accuracy']:.4f}, "
    f"F1: {res_32['eval_f1']:.4f}, Loss: {res_32['eval_loss']:.4f}"
)

print(
    f"BERTurk 128K -> Accuracy: {res_128['eval_accuracy']:.4f}, "
    f"F1: {res_128['eval_f1']:.4f}, Loss: {res_128['eval_loss']:.4f}"
)
