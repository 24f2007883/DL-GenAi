# Milestone 3 Report: Retrieval-Augmented Generation (RAG) & FAISS Vector Search

> **Objective:** Transitioning from baseline zero-shot inference to a full two-stage **Retrieval-Augmented Generation (RAG)** pipeline using FAISS dense vector search, Cross-Encoder reranking, and adversarial context analysis.

---

## Executive Summary & Key Results

| Metric / Experiment | Setup / Context | Result | Key Takeaway |
| :--- | :--- | :--- | :--- |
| **Q1: Baseline Zero-Shot** | Pure Parametric Memory (No RAG) | **0.384** (Probability) | Model shows low confidence without external context. |
| **Q2: FAISS Dense Search** | Bi-Encoder (`all-MiniLM-L6-v2`) | **Rank 10** | Retrieves answer within Top-10, but embedding similarity alone lacks precision. |
| **Q3: Cross-Encoder Rerank** | `ms-marco-MiniLM-L-6-v2` | **Rank 1** | Joint cross-attention successfully promotes true doc from Rank 10 to Rank 1. |
| **Q4: Token Budget** | Top-5 Context + Prompt | **216 Tokens** | Multi-document context easily fits standard Transformer limits (512 max). |
| **Q5: Ground-Truth Context** | Exact Correct Document Injected | **0.989** (Probability) | Ideal context boosts model confidence from 0.384 to 98.9%. |
| **Q6: Adversarial RAG** | Unrelated Document (`KB[999]`) | **0.529** (Probability) | Noise causes confidence decay (Garbage In, Garbage Out). |
| **Q7: Retrieval Hit Rate** | Top-5 FAISS Search (100 rows) | **73.0%** | FAISS finds ground-truth in Top-5 for 73/100 prompts. |
| **Q8: End-to-End Pipeline** | Two-Stage RAG Pipeline | **0.975** (MAP@3) | Full RAG pipeline yields near-perfect prediction precision. |

---

## 🛠️ System Architecture

```
The pipeline consists of a modular **Two-Stage RAG Architecture**:
[ User Query ] ──► [ Dense Encoder: MiniLM ] ──► [ FAISS Index (Top k=5) ]
                                                                  │
                                                                  ▼
[ Final Predictions ] ◄── [ Zero-Shot Classifier ] ◄── [ Cross-Encoder Reranker ]
(MAP@3 = 0.975)         (facebook/bart-large-mnli)     (ms-marco-MiniLM-L-6-v2)
```

1. **Knowledge Base (KB):** Built from 2,000 ground-truth answers extracted from `train.csv`.
2. **Dense Vector Indexing:** Documents encoded via `all-MiniLM-L6-v2` and indexed in `faiss.IndexFlatL2`.
3. **Cross-Encoder Reranking:** Computes full cross-attention over retrieved candidates to extract the single best document chunk.
4. **Context-Augmented Inference:** Passes formatted `"Context: [Best_Doc] Question: [Prompt]"` into `facebook/bart-large-mnli` for multiple-choice ranking.

---

## Detailed Experiments & Analysis

### 1. Baseline Zero-Shot vs. Ground-Truth Injection (Q1 vs. Q5)
* **Baseline (Q1):** Evaluating prompt row index 150 without context yielded a ground-truth probability of **0.384**.
* **Ideal RAG (Q5):** Injecting the exact correct KB document boosted confidence to **0.989**.
* **Insight:** External context drastically reduces model uncertainty without fine-tuning parameters.

### 2. Bi-Encoder vs. Cross-Encoder Performance (Q2 vs. Q3)
* **FAISS Retrieval (Q2):** Bi-Encoder placed the target document at **Rank 10**. Bi-Encoders map queries and passages independently, missing subtle semantic dependencies.
* **Cross-Encoder Reranking (Q3):** Cross-Encoder re-scored candidates and promoted the target document to **Rank 1** (Score: 4.7585).

### 3. Context Length & Token Budgets (Q4)
* Concatenating Top-5 retrieved documents for row index 42 generated **216 tokens** using the `bert-base-uncased` tokenizer.
* **Insight:** Keeping $k=5$ provides sufficient background information while staying well within the standard 512 token limit.

### 4. Robustness under Adversarial RAG (Q6)
* Intentionally injecting an unrelated document (`KB[999]`) caused the correct option's probability to drop to **0.529**.
* **Insight:** Demonstrates the *Garbage In, Garbage Out* risk in RAG. Reranking is critical to prevent misleading or noisy contexts from confusing the classifier.

### 5. System Retrieval Capability (Q7)
* Across the first 100 rows, FAISS achieved a **Hit Rate @ 5 of 73.0%**.
* Indicates that for 73% of queries, the correct context is present within the initial Top-5 vector search window.

---

## Final Pipeline Evaluation (Q8)

Evaluating the end-to-end pipeline across the first 20 rows using **Mean Average Precision @ 3 (MAP@3)**:

$$\text{MAP@3} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{Rank}_i}$$

* **Pipeline Pipeline Execution:**
  1. FAISS dense search retrieves $k=5$ candidate docs.
  2. Cross-Encoder scores candidates and selects $1$ best document (`np.argmax`).
  3. Prompt augmented with context string.
  4. BART MNLI outputs candidate probabilities mapped back to option letters (`A`–`E`).
* **Final Result:** **`0.975` MAP@3 Score**.

---

## Technical Stack

* **Language / Environment:** Python 3.13, Jupyter Notebook
* **Libraries:** `pandas`, `numpy`, `faiss-cpu`, `sentence-transformers`, `transformers`, `scikit-learn`
* **Models Used:**
  * `all-MiniLM-L6-v2` (Bi-Encoder Embeddings)
  * `ms-marco-MiniLM-L-6-v2` (Cross-Encoder Reranker)
  * `facebook/bart-large-mnli` (Zero-Shot Classifier)
  * `bert-base-uncased` (Tokenizer Analysis)