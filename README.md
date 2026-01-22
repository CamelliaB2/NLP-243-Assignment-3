# Language Modeling with RNNs and Transformers (Perplexity Analysis)

This project explores **language modeling** as an unsupervised NLP task. The goal is to predict the next token in a sequence given the previous tokens and to evaluate model quality using **perplexity**. Multiple architectures are compared, including RNN variants and Transformers with different positional encodings.

This project emphasizes *correct modeling setup*, *masking*, and *fair comparison across architectures*.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Problem Formulation](#problem-formulation)
4. [Evaluation Metric: Perplexity](#evaluation-metric-perplexity)
5. [Models Implemented](#models-implemented)
6. [Positional Encoding in Transformers](#positional-encoding-in-transformers)
7. [Training Setup](#training-setup)
8. [Experiments](#experiments)
9. [Results Summary](#results-summary)
10. [Critical Debugging Insight: Masking](#critical-debugging-insight-masking)
11. [Key Lessons Learned](#key-lessons-learned)
12. [Limitations](#limitations)
13. [Future Improvements](#future-improvements)
14. [Notes to Future Me](#notes-to-future-me)

---

## Project Overview

The objective of this assignment is to build a **language model** that predicts the next token in a sequence and evaluates prediction quality using **perplexity**.

Unlike prior assignments:
- This is **unsupervised learning**
- No labels are provided
- The model learns purely from token sequences

The project compares:
- Transformer-based models
- Recurrent models (LSTM, GRU, Elman RNN)
- Different positional encoding strategies
- Fine-tuned vs default hyperparameters

---

## Dataset

The dataset used is the **Penn Treebank (PTB)** dataset, accessed via Hugging Face.

### Dataset Properties
- Pre-tokenized text
- Contains `<unk>` tokens
- Already split into:
  - Train: 42,068 sentences
  - Validation: 3,370 sentences
  - Test: 3,761 sentences

Total: **49,199 sentences**

The dataset comes pre-split, reducing evaluation ambiguity.

---

## Problem Formulation

This is a **next-token prediction** task.

For each sentence:
- Input: all tokens except the last
- Target: all tokens except the first

Example:
