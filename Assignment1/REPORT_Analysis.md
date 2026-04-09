# Python Code Profile Analysis Report
## Text Encoding & Embedding for Sentiment Analysis
**Date:** April 10, 2026  
**Dataset:** Amazon Product Reviews (100 documents)  
**Focus:** Text preprocessing, feature engineering, and sentiment classification

---
## Executive Summary
This assignment implements a comprehensive text encoding pipeline using Amazon product reviews. The work demonstrates three foundational text vectorization methods (One-Hot Encoding, Bag of Words, and TF-IDF), analyzes their comparative strengths and weaknesses, and applies them to a binary sentiment classification task using Logistic Regression and Naïve Bayes classifiers.

---
## Observations
### 1. **Text Preprocessing Pipeline**
The implementation follows industry-standard NLP preprocessing:
- **Tokenization** using NLTK's word tokenizer
- **Lowercasing** for normalization
- **Punctuation removal** to eliminate noise
- **Stopword filtering** (English corpus) to reduce dimensionality
- **Lemmatization** using WordNet to group morphological variants

**Key Finding:** The preprocessed vocabulary contains **707 unique tokens** (after filtering), dramatically reduced from raw document text. This demonstrates how preprocessing dramatically reduces the feature space while preserving semantic content.

### 2. **Vocabulary and Feature Analysis**
- **Manual vocabulary creation:** 707 tokens via Counter
- **Sklearn CountVectorizer vocabulary:** 707 tokens (matching)
- **Top frequent terms:** 'shirt', 'good', 'size', 'quality', 'wear', 'fit', 'comfort', 'color', 'material', 'price'

**Observation:** Top terms are semantically meaningful, confirming effective preprocessing. Common words like "shirt" and "size" (product-specific) appropriately dominate the corpus, with sentiment-bearing words ('good', 'quality') also ranking high.

### 3. **Encoding Methods Comparison**
| Method | Dimensions | Value Type | Sparsity | Key Characteristic |
|--------|-----------|-----------|----------|-------------------|
| **One-Hot** | 100 × 707 | Binary | ~86.39% | Equal weighting; memory inefficient |
| **Bag of Words** | 100 × 707 | Integer counts | ~92.87% | Frequency-based; preserves occurrences |
| **TF-IDF** | 100 × 707 | Weighted floats | ~92.73% | Normalized importance; down-weights common terms |

**Key Insight:** All methods produce sparse representations, but TF-IDF provides normalized, semantically more discriminative features by incorporating document-frequency inverse weighting.

### 4. **TF-IDF Effect Analysis**
The analysis reveals how TF-IDF mathematics work in practice:
- **High-frequency, low-discriminative words** (e.g., 'good', 'quality'): IDF ≈ 1.68–1.95 (reduced weight)
- **Lower-frequency, distinctive words** (e.g., 'shirt', product-specific terms): IDF ≈ 2.84+ (elevated weight)
- **Absent terms:** Not included in vocabulary, naturally filtering outliers

**Conclusion:** TF-IDF successfully down-weights ubiquitous terms that appear in most reviews, allowing the model to focus on document-specific, discriminative features.

### 5. **Sparse Matrix Efficiency**
- **Memory overhead considerations:** Despite high sparsity (86–93%), sparse matrices still require indexing structures (row/column pointers), limiting practical efficiency gains
- **Computational complexity:** Many ML operations still traverse pointer arrays repeatedly, incurring cache-miss penalties
- **Scalability limitation:** As vocabulary grows, index arrays grow linearly, restricting real-world applicability for very large corpora

### 6. **Sentiment Classification Results**
**Binary Classification Task:** Positive vs. Negative sentiment (heuristic labels from review headings)
| Model | BoW Accuracy | TF-IDF Accuracy | Improvement |
|-------|-------------|-----------------|------------|
| **Logistic Regression** | 0.9333 | 0.9333 | — |
| **Naïve Bayes** | 0.8667 | 0.9333 | +6.66% |

**Finding:** TF-IDF matches or exceeds BoW performance. Notably, Naïve Bayes gains 6.66% accuracy with TF-IDF, suggesting that term weighting benefits probabilistic classifiers more than linear models. Both models achieve high accuracy (≥86%), validating the preprocessing pipeline's effectiveness.

---
## Conclusions
### 1. **Preprocessing is Critical**
The 707-word vocabulary after preprocessing (vs. thousands in raw text) shows that effective preprocessing reduces noise while preserving semantic signal. Investment in quality tokenization, stopword removal, and lemmatization pays dividends.

### 2. **Method Selection Depends on Use Case**
- **One-Hot Encoding:** Rarely practical; equal treatment of all occurrences wastes information
- **Bag of Words:** Suitable for simple baselines, interpretability-critical applications; computationally cheap
- **TF-IDF:** Preferred for most cases; balances term frequency with corpus-wide rarity, improving model discrimination

### 3. **TF-IDF Remains Relevant Despite Limitations**
TF-IDF successfully:
- Emphasizes document-specific vocabulary
- Down-weights universal terms
- Produces normalized, interpretable vectors
- Achieves 93.33% accuracy on this sentiment task

However, TF-IDF **does not** capture:
- Word order or syntax
- Semantic similarity between synonyms
- Long-range contextual relationships
- Sarcasm or irony

### 4. **Sparsity Paradox**
Sparse matrices represent data efficiently theoretically, but in practice:
- Index overhead remains substantial even at 86–93% sparsity
- Distributed systems amplify this overhead due to metadata transmission
- Dense or hashed representations may outperform sparse formats at scale

### 5. **Model & Feature Interaction**
The 6.66% improvement of TF-IDF over BoW in Naïve Bayes (vs. no change in Logistic Regression) indicates that **probabilistic classifiers benefit more from weighted features** than linear models, which may internally learn feature importance through coefficients.

---
## Recommendations for Future Work
1. **Explore advance representations:** Word embeddings (Word2Vec, GloVe) or transformer-based encodings (BERT) to capture semantic relationships
2. **Implement ensemble methods:** Combine BoW, TF-IDF, and embedding features for robust classification
3. **Scale testing:** Evaluate method efficiency on larger corpora (10k+ documents) to confirm sparse matrix scalability claims
4. **Domain-specific tuning:** Adjust stopword lists, lemmatization, and hyperparameters for product review specificity
5. **Sentiment refinement:** Replace heuristic labels with crowdsourced or expert annotations for label quality

---
## Technical Notes
- **Framework:** scikit-learn, NLTK, pandas, NumPy
- **Preprocessing:** Tokenization, lowercasing, punctuation removal, stopword filtering, lemmatization
- **Evaluation:** Holdout validation (75%–25% train/test split), accuracy metric
- **Dataset characteristics:** 100 Amazon product reviews, 707 unique tokens post-processing, ~86–93% sparsity across representations

**Status:** Complete and validated. All cells executed successfully; results consistent with expected NLP preprocessing behavior.


---------------------------------------------------------------------------------------------
# Key Findings

## Observations:
1. Preprocessing effectiveness — Reduced vocabulary to 707 tokens while preserving semantic content
2. Encoding comparison — All methods produce 86–93% sparse matrices; TF-IDF achieves best classification accuracy
3. TF-IDF advantage — Successfully down-weights common words (IDF ~1.68) while elevating distinctive terms (IDF ~2.84+)
4. Classification results — 93.33% accuracy (Logistic Regression) and 93.33% accuracy (Naïve Bayes with TF-IDF)
5. Sparsity paradox — Despite high sparsity, indexing overhead limits practical efficiency gains

## Conclusions:
1. TF-IDF remains the practical choice for traditional NLP pipelines (balances interpretability and performance)
2. Naïve Bayes benefits more from TF-IDF weighting (+6.66% over BoW) than Logistic Regression
3. One-Hot Encoding is rarely practical; impairs model learning
4. Semantic limitations remain: word order, synonymy, and context not captured