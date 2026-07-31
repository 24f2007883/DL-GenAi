# Milestone 2: Technical Summary & Evaluation Report

## Executive Overview
The primary objective of Milestone 2 is to transition from classical NLP techniques (TF-IDF, n-grams) to modern Deep Learning architectures using Transformers. This report documents theoretical insights, architectural mechanics, contextual embeddings, zero-shot classification dynamics, and small language model (SLM) generation evaluated on the `train.csv` dataset.

---

## 1. Introduction to Hugging Face Ecosystem & Preprocessing

### Question 1: Dataset Concatenation & String Length
* **Objective:** Concatenate `prompt` and `A` text columns using Hugging Face `datasets.map()`.
* **Execution:** Processed dataset without pandas overhead.
* **Result (Index 51):**
  * **Exact String Length (`len()`):** `614` characters.

### Question 2: Vocabulary Configuration (`bert-base-uncased`)
* **Objective:** Inspect vocabulary configuration properties of `AutoTokenizer`.
* **Result:**
  * **Total Vocabulary Size:** `30,522` subword tokens.

### Question 3: Special Tokens & Boundary Markers
* **Objective:** Extract structural boundary token IDs for BERT architecture.
* **Results:**
  * `[CLS]` Token ID: `101`
  * `[SEP]` Token ID: `102`
  * `[PAD]` Token ID: `0`
  * `[UNK]` Token ID: `100`
  * `[MASK]` Token ID: `103`

### Question 4: Batch Tokenization & Tensor Shapes
* **Objective:** Tokenize full dataset prompts with `max_length=128`, `padding='max_length'`, `truncation=True`.
* **Result:**
  * **`input_ids` Shape:** `torch.Size([2000, 128])`

---

## 2. BERT Architecture & Internal Attention Mechanics

### Question 6 & 7: Last Hidden State & [CLS] Vector Representation
* **Objective:** Pass Row 0 prompt through `bert-base-uncased` to inspect embedding dimensions and extract the `[CLS]` pooling representation.
* **Results:**
  * **`last_hidden_state` Shape:** `torch.Size([1, 31, 768])` (Batch = 1, Sequence Length = 31, Hidden Dim = 768)
  * **[CLS] Vector First 5 Sum:** `-1.2001` (First 5 values: `[-0.4677, -0.0754, -0.2019, -0.0071, -0.4480]`)

### Question 8: Self-Attention Weight Extraction
* **Objective:** Analyze self-attention tensor for input `"Light-ion fusion is a technique."` across Layer -1, Head 0.
* **Token Mapping:** Index 0: `[CLS]`, Index 1: `light`, Index 2: `-`, Index 3: `ion`, Index 4: `fusion`, Index 5: `is`, Index 6: `a`, Index 7: `technique`, Index 8: `.`, Index 9: `[SEP]`.
* **Result:**
  * **Attention Weight (`[CLS]` $\rightarrow$ `fusion` at `[0, 4]`):** `0.1025`

---

## 3. Context-Aware Sentence Embeddings & Ranking Pipelines

### Question 9: Dense Vector Cosine Similarity
* **Objective:** Calculate cosine similarity between Row 0 `prompt` and `Option B` using `sentence-transformers/all-MiniLM-L6-v2`.
* **Vector Dimension:** `torch.Size([384])`
* **Result:**
  * **Cosine Similarity Score:** `0.7658`

### Question 10: Comparative Pipeline Evaluation (TF-IDF vs MiniLM)
* **Objective:** Evaluate MAP@3 performance across all 2,000 dataset rows and count ranking improvements.
* **Results:**
  * **`all-MiniLM-L6-v2` MAP@3 Score:** `0.4231`
  * **Questions Recovered (Not in TF-IDF Top-3 BUT in MiniLM Top-3):** `502`

| Model / Pipeline | Overall MAP@3 | Key Mechanism |
| :--- | :--- | :--- |
| **TF-IDF Cosine Similarity** | ~0.2962 | Exact Keyword Overlap (Lexical Search) |
| **Sentence-MiniLM-L6-v2** | ~0.4231 | Dense Contextual Vector Alignment (Semantic Search) |

> **Key Insight:** Dense embeddings captured semantic meaning in **502 questions** where TF-IDF failed due to zero keyword overlap.

---

## 4. Zero-Shot Classification & Decision Functions

### Question 11: Single-Label Classification (Softmax)
* **Objective:** Evaluate `facebook/bart-large-mnli` zero-shot pipeline on Row 1 prompt with Options A, B, C.
* **Result:**
  * **Top Option Confidence Score:** `0.4575`

### Question 12: Multi-Label Probability Analysis (Softmax vs. Sigmoid)
* **Objective:** Compare candidate score sums under Softmax (`multi_label=False`) versus independent Sigmoid (`multi_label=True`).
* **Results:**
  * **Softmax Probability Sum:** `1.0000`
  * **Sigmoid Probability Sum:** `0.0005`
  * **Absolute Sum Difference:** `0.9995`

---

## 5. Generative AI with Small Language Models (SLMs)

### Question 13: Instruction-Tuned Text Generation (`google/flan-t5-small`)
* **Objective:** Format Row 0 prompt into an explicit A/B decision query for `google/flan-t5-small`.
* **Execution Parameters:** `max_new_tokens=5`, `skip_special_tokens=True`.
* **Result:**
  * **Generated Output String:** `"B"`

---

## Conclusion
Milestone 2 validates the superiority of Transformer-based dense representations over classical n-gram approaches, demonstrating high contextual precision, zero-shot flexibility, and direct instruction execution with lightweight SLMs.