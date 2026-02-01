# LVC Classification – Lemma vs Syntax vs BERTurk

This repository contains minimal reference implementations used in the paper:

**From Lemmas to Dependencies: What Signals Drive Light Verb Construction Classification?**

The code compares three modeling settings:
- Lemma-only (TF-IDF + Logistic Regression)
- Syntax-only (UPOS + DEPREL + MORPH counts + Logistic Regression)
- BERTurk fine-tuning (32k vs 128k)
