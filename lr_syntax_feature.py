import pandas as pd
from conllu import parse_incr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

TRAIN_PATHS = [
    "compaund_lightverb_merged_labeled.conllu",
    "compound_merged_labeled.conllu"
]

TEST_PATH = "test_parsed_labeled.conllu"

MODEL_PATH = "lr_syntax_model.pkl"


def collect_feature_vocab(conllu_paths):
    upos_set = set()
    deprel_set = set()
    morph_set = set()

    for path in conllu_paths:
        with open(path, "r", encoding="utf-8") as f:
            for sent in parse_incr(f):
                for tok in sent:
                    if tok["upos"]:
                        upos_set.add(f"UPOS_{tok['upos']}")
                    if tok["deprel"]:
                        deprel_set.add(f"DEPREL_{tok['deprel']}")
                    if tok["feats"]:
                        for k, v in tok["feats"].items():
                            morph_set.add(f"MORPH_{k}={v}")

    features = sorted(upos_set | deprel_set | morph_set)
    return features

FEATURES = collect_feature_vocab(TRAIN_PATHS)

print(f"Toplam syntax feature sayısı: {len(FEATURES)}")

# =========================
# 2) FEATURE EXTRACTOR
# =========================

def extract_features(sent, FEATURES):
    feats = dict.fromkeys(FEATURES, 0)

    for tok in sent:
        if tok["upos"]:
            k = f"UPOS_{tok['upos']}"
            if k in feats:
                feats[k] += 1

        if tok["deprel"]:
            k = f"DEPREL_{tok['deprel']}"
            if k in feats:
                feats[k] += 1

        if tok["feats"]:
            for mk, mv in tok["feats"].items():
                k = f"MORPH_{mk}={mv}"
                if k in feats:
                    feats[k] += 1

    return feats


def load_train(paths, FEATURES):
    X, y = [], []

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for sent in parse_incr(f):
                X.append(extract_features(sent, FEATURES))
                y.append(int(sent.metadata["label"]))  # 1 = LVC

    return pd.DataFrame(X), pd.Series(y)

X_train, y_train = load_train(TRAIN_PATHS, FEATURES)

print("TRAIN SHAPE:", X_train.shape)
print("TRAIN LABEL DAĞILIMI:\n", y_train.value_counts())


model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

joblib.dump(model, MODEL_PATH)

print("Model eğitildi ve kaydedildi.")

def load_test(path, FEATURES):
    X, y = [], []

    with open(path, "r", encoding="utf-8") as f:
        for sent in parse_incr(f):
            X.append(extract_features(sent, FEATURES))
            y.append(int(sent.metadata["label"]))

    return pd.DataFrame(X), pd.Series(y)

X_test, y_test = load_test(TEST_PATH, FEATURES)

print("TEST SHAPE:", X_test.shape)
print("TEST LABEL DAĞILIMI:\n", y_test.value_counts())

model = joblib.load(MODEL_PATH)

y_pred = model.predict(X_test)

print("\n===== LR-SYNTAX TEST RAPORU =====\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=["NLVC", "LVC"]
))
