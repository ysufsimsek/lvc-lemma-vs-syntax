import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight



TRAIN_CSV_1 = "sentences_with_lemmas_and_labels.csv"
TRAIN_CSV_2 = "sentences_with_lemmas_and_labels_lightveb.csv"

PREDICT_CSV = "lvc_with_sentence_lemma.csv"          
PREDICT_COL = "sentence_lemma"                      

OUT_CSV = "lvc_same_words_with_predictions.csv"

RANDOM_SEED = 42
TEST_SIZE = 0.2


NGRAM_RANGE = (1, 2)
MAX_FEATURES = 5000
LOWERCASE = False   



df1 = pd.read_csv(TRAIN_CSV_1)
df2 = pd.read_csv(TRAIN_CSV_2)
df = pd.concat([df1, df2], ignore_index=True)

if "lemmas" not in df.columns or "label" not in df.columns:
    raise ValueError("Training CSVs must contain columns: 'lemmas' and 'label'.")

X = df["lemmas"].fillna("").astype(str)
y = df["label"].astype(int)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y
)

print("Train class counts:", y_train.value_counts().to_dict())
print("Test  class counts:", y_test.value_counts().to_dict())


classes = np.array([0, 1])
w = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
print("Computed class weights (balanced):", dict(zip(classes, w)))


vectorizer = TfidfVectorizer(
    ngram_range=NGRAM_RANGE,
    max_features=MAX_FEATURES,
    lowercase=LOWERCASE
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)


model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)
model.fit(X_train_tfidf, y_train)


y_pred_default = model.predict(X_test_tfidf)

print("\n=== Default threshold (0.50) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_default))
print(classification_report(y_test, y_pred_default, digits=3))
print("Confusion matrix [ [TN FP] [FN TP] ]:\n", confusion_matrix(y_test, y_pred_default))


proba_test = model.predict_proba(X_test_tfidf)[:, 1]
prec, rec, thr = precision_recall_curve(y_test, proba_test)

f1 = 2 * prec * rec / (prec + rec + 1e-12)


best_i = int(np.argmax(f1[:-1])) 
best_thr = float(thr[best_i])

print("\n=== Tuned threshold (best F1) ===")
print("Best threshold:", best_thr)
y_pred_tuned = (proba_test >= best_thr).astype(int)
print("Accuracy:", accuracy_score(y_test, y_pred_tuned))
print(classification_report(y_test, y_pred_tuned, digits=3))
print("Confusion matrix [ [TN FP] [FN TP] ]:\n", confusion_matrix(y_test, y_pred_tuned))


"""
target_precision = 0.70
valid = np.where(prec[:-1] >= target_precision)[0]
if len(valid) > 0:
    i = valid[-1]              # highest recall under target precision
    thr_p = float(thr[i])
    print(f"\n=== Tuned threshold (precision >= {target_precision}) ===")
    print("Threshold:", thr_p)
    y_pred_p = (proba_test >= thr_p).astype(int)
    print(classification_report(y_test, y_pred_p, digits=3))
"""


predict_df = pd.read_csv(PREDICT_CSV)

if PREDICT_COL not in predict_df.columns:
    raise ValueError(f"'{PREDICT_COL}' column not found in {PREDICT_CSV}.")

predict_text = predict_df[PREDICT_COL].fillna("").astype(str)
predict_tfidf = vectorizer.transform(predict_text)


predict_proba = model.predict_proba(predict_tfidf)[:, 1]
predict_df["predicted_proba"] = predict_proba
predict_df["predicted_label_default"] = (predict_proba >= 0.50).astype(int)
predict_df["predicted_label_tuned"]   = (predict_proba >= best_thr).astype(int)

predict_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
print(f"\n✅ Predictions saved to: {OUT_CSV}")
print("Columns added: predicted_proba, predicted_label_default, predicted_label_tuned")
