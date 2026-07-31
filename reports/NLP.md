# Milestone 1 Report: Exploratory Data Analysis, Text Preprocessing, and Baseline Similarity Pipeline

## 1. Executive Summary
The primary objective of Milestone 1 was to establish a foundational Natural Language Processing (NLP) pipeline for a multiple-choice question-answering dataset. This involved performing Exploratory Data Analysis (EDA), cleaning raw text through standardization and stop-word removal, building term-based feature representations using TF-IDF vectorization, and calculating baseline evaluation metrics (Accuracy and MAP@3).

Our findings indicate that simple lexical similarity (TF-IDF + Cosine Similarity) yields a top-choice accuracy of **13.55%** and a MAP@3 score of **0.29617**, which underperforms compared to a simple Majority Class Baseline (MAP@3 of **0.42125**). This highlights the necessity of dense, semantic-aware embeddings (e.g., Transformer models) for subsequent milestones.

---

## 2. Methodologies & Preprocessing Steps
To convert unstructured text into actionable numerical features, a systematic four-step processing pipeline was applied:

1. **Text Normalization:** Converted all characters to lowercase to eliminate case-sensitivity discrepancies.
2. **Punctuation Removal:** Stripped standard punctuation marks using Python's `string.punctuation` via efficient mapping tables (`str.maketrans`).
3. **Tokenization & Stop-Word Filtering:** Split text on whitespace boundaries into individual tokens and removed non-informative functional words using `sklearn.feature_extraction.text.ENGLISH_STOP_WORDS`.
4. **TF-IDF Vectorization:** Transformed combined textual inputs (prompts and options A through E) into sparse numerical feature vectors using `TfidfVectorizer(stop_words='english')`.
5. **Similarity Calculation:** Computed pairwise Cosine Similarity between the prompt vector and each candidate option vector to rank candidate answers from highest to lowest similarity.

---

## 3. Key Observations & Results

| Metric / Observation | Quantitative Value | Interpretation & Technical Takeaway |
| :--- | :--- | :--- |
| **Dataset Shape** | Train: 2,000 rows × 8 cols<br>Test: 500 rows × 7 cols | Compact dataset structure with zero missing values across both train and test splits. |
| **Target Distribution** | B: 490, C: 459, A: 369,<br>D: 358, E: 324 | Option B is the most frequent target answer, while E is the least frequent. Target classes are reasonably balanced. |
| **Prompt Vocabulary Size** | 859 unique words | Total unique words remaining across all cleaned prompts after lowercasing and punctuation stripping. |
| **Row ID 1 Filtered Tokens** | 13 unique tokens | Removing standard English stop words reduced prompt noise to 13 high-information keywords. |
| **Combined TF-IDF Features** | 2,762 features | Fitting `TfidfVectorizer` across all prompts and options (A–E) generated a 2,762-dimensional feature space. |
| **Row ID 1 Cosine Similarity (Prompt vs Option A)** | 0.2720 | Calculated baseline cosine distance for Row ID 1 Option A using TF-IDF feature vectors. |
| **TF-IDF Top-1 Accuracy** | 13.55% | Percentage of instances where the option with the absolute highest cosine similarity matched the ground truth answer. |
| **Majority Class Baseline (MAP@3)** | 0.42125 | MAP@3 score achieved by statically predicting the top 3 most frequent overall choices (B, C, A) for all rows. |
| **TF-IDF Pipeline Baseline (MAP@3)** | 0.29617 | MAP@3 score achieved by dynamically ranking options based on TF-IDF cosine similarity scores. |

---

## 4. Discussion & Limitations
The primary finding from Milestone 1 is that the **Majority Class Baseline (0.42125 MAP@3)** outperforms the **TF-IDF Cosine Similarity Pipeline (0.29617 MAP@3)**.

This result provides critical insight into the problem domain:
* **Lexical vs. Semantic Gap:** TF-IDF measures surface-level word overlap (term frequency-inverse document frequency). Multiple-choice questions frequently rephrase concepts using synonyms or require logical reasoning, where correct options share few direct words with the prompt.
* **Spurious Keyword Matches:** Distractor options often include identical technical terminology from the prompt, artificially inflating their TF-IDF cosine similarity score without being logically correct.

---

## 5. Conclusion & Next Steps
Milestone 1 successfully established our benchmarking environment, validation metric (MAP@3), and preprocessed data pipelines. Because sparse lexical representations proved insufficient for complex multiple-choice reasoning, future milestones will transition to dense semantic representations, utilizing contextual embeddings and fine-tuned Transformer architectures (e.g., BERT, DeBERTa, or LLM-based approaches).