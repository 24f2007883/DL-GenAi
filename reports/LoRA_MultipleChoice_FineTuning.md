# Milestone 4: Multiple-Choice Classification & Parameter-Efficient Fine-Tuning (LoRA)

## Executive Summary
Milestone 4 focuses on formulating the Smart MCQ Solver Challenge as a proper multiple-choice classification problem. This phase transitions from zero-shot and generic pipelines to training `AutoModelForMultipleChoice` models using Low-Rank Adaptation (LoRA) and Hugging Face's `Trainer` API.

### Core Architecture Pillars
1. **Data Formatting**: Transforming single questions with 5 options into 5 separate `[Prompt + SEP + Option]` pairs.
2. **3D Tensor Structures**: Structuring inputs into `[Batch Size, Num Choices, Sequence Length]` tensors expected by multiple-choice architectures.
3. **PEFT / LoRA Integration**: Freezing base transformer weights and fine-tuning adapter matrices (~0.27% trainable parameters).
4. **Trainer & Inference Pipeline**: Executing fine-tuning loops and applying Softmax to derive calibrated option probabilities.

---

## Technical Implementations & Questions Summary

### Question 1: Label Encoding

The categorical answers (A--E) were converted into numerical labels
(0--4). This encoding is required because deep learning models are
trained using integer class labels rather than text labels.

### Question 2: Prompt-Option Formatting

Each answer option was paired individually with the original prompt
using the format: `prompt + " [SEP] " + option` This produces one
independent input sequence for every answer choice.

### Question 3: Single-Row Tokenization

The five formatted prompt-option pairs were tokenized using
`bert-base-uncased` with a maximum sequence length of 128. The resulting
tensor followed the required multiple-choice format of
`[batch_size, num_choices, sequence_length]`.

### Question 4: Batch Tokenization

The first sixteen training samples were converted into batches of
multiple-choice inputs. Each sample contained five choices, each padded
or truncated to 128 tokens.

### Question 5: Multiple-Choice Model Outputs

`AutoModelForMultipleChoice` generated one logit score for each answer
option. These logits represent the model's confidence before Softmax
normalization.

### Question 6: Supervised Training Loss

The correct encoded label was supplied to the model during the forward
pass. The returned loss tensor was a scalar, making it suitable for
optimization during training.

### Question 7: LoRA Fine-Tuning

Low-Rank Adaptation (LoRA) was applied to the BERT multiple-choice
model. Only lightweight adapter parameters remained trainable while the
original model weights stayed frozen, significantly reducing training
cost.

### Question 8: Hugging Face Dataset Preparation

The first one hundred training samples were converted into a Hugging
Face Dataset containing: - input_ids - attention_mask - labels

Each dataset item stored five tokenized answer choices.

### Question 9: Tiny Fine-Tuning

A small LoRA fine-tuning experiment was performed using the Hugging Face
Trainer on thirty-two training examples with four optimization steps.
This verified that the complete training pipeline executed successfully.

### Question 10: Probability Prediction

After fine-tuning, inference was performed on one question. The output
logits were converted into probabilities using the Softmax function,
producing a probability distribution across all five answer options.

## Key Learning Outcomes

-   Understood how multiple-choice classification differs from standard
    text classification.
-   Learned how to prepare prompt-option pairs for transformer models.
-   Practiced tokenization for multiple-choice architectures.
-   Used `AutoModelForMultipleChoice` for inference and supervised
    learning.
-   Applied LoRA for parameter-efficient fine-tuning.
-   Prepared datasets compatible with the Hugging Face Trainer.
-   Performed a complete training and inference workflow.
-   Converted logits into probabilities using Softmax.

## Conclusion

This milestone demonstrated the complete pipeline for transformer-based
multiple-choice question answering. Starting from raw Kaggle data, the
dataset was transformed into model-ready inputs, fine-tuned efficiently
using LoRA, and evaluated through probability-based predictions. The
workflow provides a practical foundation for building efficient large
language model solutions for multiple-choice reasoning tasks.
