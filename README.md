# Lemma vs. Syntax for Light Verb Construction Classification

This repository contains the data and code for the paper:

**From Lemmas to Dependencies: What Signals Drive Light Verb Construction Classification?**

---

## 🔍 TL;DR

This study investigates **which linguistic signals drive Light Verb Construction (LVC) classification in Turkish**.

- Lemma-based models already achieve strong performance.
- Syntax-only models using dependency features are competitive but sensitive to parsing noise.
- BERTurk models achieve the best overall results, but much of their success comes from lexical memorization.
- Lemma-only Transformer models reveal distribution shift effects when evaluated on surface-form inputs.

---

## 📁 Content

### 📦 Code

The repository provides minimal, experiment-oriented scripts:

- `lemma_tfidf_lr.py`  
  TF-IDF features over lemma n-grams with Logistic Regression.

- `lr_syntax_feature.py`  
  Syntax-only classification using Universal Dependencies features:
  UPOS tags, dependency relations, and morphological attributes.

- `BERTurk32k-128k.py`  
  Supervised fine-tuning of BERTurk models (32k and 128k vocabularies)
  using surface-form inputs.

- `lemma_BERTurk32k-128K.py`  
  Fine-tuning BERTurk using **lemmatized token sequences** as model input.

---

## 🧠 Datasets

### Universal Dependencies Turkish Treebanks

Light Verb Constructions used for training are extracted from the following
**Universal Dependencies Turkish treebanks**:

- UD Turkish-ATIS  
- UD Turkish-BOUN  
- UD Turkish-FrameNet  
- UD Turkish-GB  
- UD Turkish-IMST  
- UD Turkish-Kenet  
- UD Turkish-PUD  
- UD Turkish-Penn  
- UD Turkish-Tourism  

These treebanks cover multiple domains (spoken language, news, travel, etc.),
enabling robust evaluation across diverse linguistic contexts.

---

### Diagnostic Evaluation Set

This repository **includes a manually curated diagnostic test set**:

**`lvc_same_words_with_labels2.csv`**

The dataset consists of **147 Turkish sentences**, balanced across three
evaluation categories (**49 instances each**):

- **LVC Positive**  
  Idiomatic light verb constructions.

- **NLVC (Near-LVC)**  
  Literal uses that share the same target verb–noun components.

- **Random Negatives**  
  In-domain random negative control sentences.

#### File Format

| Column        | Description |
|--------------|-------------|
| `sentence`   | Turkish sentence |
| `label`      | Binary label (1 = LVC, 0 = non-LVC) |
| `etiket_turu`| Evaluation category (LVC / NLVC / Random) |

This diagnostic set is designed to **explicitly probe model decision boundaries**
and to distinguish lexical memorization from deeper linguistic generalization.

---

## 📊 Evaluation

Experiments are organized into four modeling regimes:

### Task 1: Lemma-Only Classification
Supervised learning using lemma n-grams with TF-IDF and Logistic Regression.

### Task 2: Syntax-Only Classification
Bag-of-features representation derived from dependency parses
(UPOS, DEPREL, MORPH).

### Task 3: Supervised Transformer Models
Fine-tuning BERTurk (32k and 128k) using surface-form inputs.

### Task 4: Lemma-Based Transformer Models
Fine-tuning BERTurk using lemmatized token sequences
to isolate lexical effects.

Evaluation emphasizes **split-wise performance** rather than single aggregate scores.

---

## 🔁 Reproducibility

- Fixed random seeds across experiments
- Stratified train/test splits
- Multiple runs for Transformer-based models
- Diagnostic evaluation reported separately

---

## 📌 Key Findings

- Lexical information explains a large portion of LVC classification performance.
- Syntax contributes useful signals but is fragile under parsing errors.
- Transformer models rely heavily on memorized verb–noun associations.
- Lemma-only Transformers expose robustness issues under input mismatch.

---

## 📖 Citation

If you use this code or dataset, please cite:

> **From Lemmas to Dependencies: What Signals Drive Light Verb Construction Classification?**  
> Yusuf Şimşek, Sercan Karakaş

---

## 📬 Contact

- **Yusuf Şimşek** — ysimsek@firat.edu.tr  
- **Sercan Karakaş** — skarakas@uchicago.edu  

---

## 📄 License

License information will be added.
