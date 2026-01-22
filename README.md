# Language Modeling and Perplexity Evaluation (RNNs vs Transformers)

This project explores **language modeling** as an unsupervised NLP task. The goal is to predict the next token in a sequence given the previous tokens and evaluate model quality using **perplexity**. Multiple architectures are compared, including RNN variants and Transformers with different positional encodings.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Task Definition](#task-definition)
4. [Evaluation Metric: Perplexity](#evaluation-metric-perplexity)
5. [Models Implemented](#models-implemented)
6. [Training Setup](#training-setup)
7. [Experiments](#experiments)
8. [Results Summary](#results-summary)
9. [Key Lessons Learned](#key-lessons-learned)
10. [Critical Mistakes and Fixes](#critical-mistakes-and-fixes)
11. [Limitations](#limitations)
12. [Future Improvements](#future-improvements)
13. [Notes to Future Me](#notes-to-future-me)

---

## Project Overview

This assignment focuses on **next-token prediction**, a core problem in language modeling. Given a sequence of tokens, the model predicts the probability distribution over the next token.

Unlike previous assignments:
- There are **no labels**
- The task is **unsupervised**
- Model quality is evaluated via **perplexity**, not accuracy

The objective is to compare different neural architectures and determine which produces the lowest perplexity on held-out data.

---

## Dataset

The dataset used is the **Penn Treebank (PTB)** dataset, loaded from Hugging Face.

### Dataset Properties
- Total sentences: **49,199**
- Pre-split into:
  - Train: 42,068
  - Validation: 3,370
  - Test: 3,761

The dataset contains tokenized sentences with:
- `<unk>` tokens
- Placeholder numbers (`N`)
- Financial/news-style language

Because the dataset is pre-split, no custom splitting is required.

---

## Task Definition

### Input
- A sequence of tokens:  
  `t₁, t₂, ..., tₙ₋₁`

### Output
- Probability distribution over the next token:  
  `p(tₙ | t₁...tₙ₋₁)`

Training setup:
- Inputs: all tokens except the last
- Targets: all tokens except the first

This formulation allows the model to learn conditional probabilities for token prediction.

---

## Evaluation Metric: Perplexity

Perplexity measures how well a language model predicts a sequence.

\[
\text{Perplexity} = \exp\left(-\frac{1}{T} \sum_{i=1}^{T} \log p(t_i | t_{<i}) \right)
\]

Key interpretation:
- **Lower is better**
- Perplexity = 1 is perfect prediction
- High perplexity = high uncertainty

Perplexity is sensitive to:
- Data leakage
- Masking errors
- Model architecture

---

## Models Implemented

### Transformer Models
- Transformer with **learned positional encoding**
- Transformer with **sinusoidal positional encoding**

Why positional encoding matters:
- Transformers have no recurrence or convolution
- Token order must be injected explicitly

Two encoding strategies were compared to measure their impact on perplexity.

---

### Recurrent Models
- **LSTM**
- **GRU**
- **Elman RNN**

These models inherently capture sequential order via recurrence and serve as strong baselines for language modeling.

---

## Training Setup

### Optimizer
- Adam (with weight decay)

### Loss Function
- CrossEntropyLoss

### Hyperparameters Explored
- `d_model`
- `n_head`
- `n_layer`
- `batch_size`
- `learning_rate`
- `dropout`

### Hyperparameter Optimization
- **Optuna** used
- 20 trials
- 5 epochs per trial
- Final models trained for 20 epochs

Shorter training was used due to runtime constraints.

---

## Experiments

### Experiment 1: Transformer (Default)
- d_model: 256
- n_head: 8
- n_layer: 2
- learning_rate: 0.001
- Avg perplexity: **312.84**

---

### Experiment 2: Transformer + Sinusoidal Encoding
- Same parameters as Experiment 1
- Avg perplexity: **303.71**

This showed a small but consistent improvement over learned positional encoding.

---

### Experiment 3: LSTM
- Avg perplexity: **359.12**
- Higher loss and worse perplexity than Transformers

---

### Experiment 4: GRU
- Avg perplexity: **315.18**
- Best-performing RNN variant

---

### Experiment 5: Elman RNN
- Avg perplexity: **342.07**
- Performed worse than GRU and Transformers

---

### Experiment 6: Fine-Tuned Transformer (Best Model)
- Transformer + sinusoidal encoding
- Optuna-tuned hyperparameters:
  - n_head: 4
  - n_layer: 4
  - learning_rate: ~0.00054
  - dropout: 0.2
- Avg perplexity: **298.09**

This was the **best overall result**.

---

## Results Summary

| Model | Avg Perplexity |
|-----|----------------|
| Transformer (default) | 312.84 |
| Transformer (sin/cos) | 303.71 |
| LSTM | 359.12 |
| GRU | 315.18 |
| Elman RNN | 342.07 |
| Fine-tuned Transformer | **298.09** |

---

## Key Lessons Learned

- Transformers outperform RNNs for language modeling on PTB
- Sinusoidal positional encoding slightly outperforms learned encoding
- GRU is the strongest RNN baseline
- Hyperparameter tuning matters more than optimizer choice
- Learning rate stability is critical
- Lower perplexity requires correct **causal masking**

---

## Critical Mistakes and Fixes

### ❌ Missing Causal Masking (Initial Experiments)
Early experiments accidentally allowed the model to:
- See future tokens
- Artificially lower perplexity (down to the 100s)

### ✅ Fix
- Implemented proper **causal masking**
- Results became realistic and trustworthy

This mistake was instructional:
> Transformers without masking behave more like bidirectional models (e.g., BERT), which is inappropriate for next-token prediction but useful for other NLP tasks.

---

## Limitations

- No large-scale transformer models
- Limited training epochs
- No subword modeling beyond PTB tokenization
- Evaluation focused primarily on average perplexity

---

## Future Improvements

If revisiting this project:
1. Increase training duration
2. Use larger transformer architectures
3. Add learning rate scheduling
4. Compare against pretrained LMs
5. Evaluate perplexity per sentence length
6. Add qualitative text generation examples

---

## Notes to Future Me

This project reinforced that:
- **Correct masking is non-negotiable in language modeling**
- Small implementation details can completely invalidate metrics
- Transformers shine in autoregressive settings when used correctly
- Perplexity is powerful but unforgiving

The biggest takeaway was not the final score — it was understanding *why* incorrect setups can look deceptively good.
