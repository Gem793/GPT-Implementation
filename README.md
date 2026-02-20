Building GPT from Scratch: A Character-Level Language Model
===========================================================

This project builds a Generatively Pre-trained Transformer (GPT) from scratch. The implementation is a decoder-only Transformer designed to model sequences at the character level.

The goal is to train a model capable of generating infinite text in a specific style. While professional systems like ChatGPT are trained on massive datasets, this project uses the Tiny Shakespeare dataset—a 1MB file containing the complete works of Shakespeare.

Project Overview
----------------
- Train a character-level language model.
- Generate text in the style of Shakespeare.
- Learn the core concepts of GPT through a simple, educational implementation.

Core Architecture
-----------------
- Tokenization: Converts raw text into integers using a character-level tokenizer (vocabulary size: 65).
- Self-Attention Mechanism: Tokens communicate via Queries, Keys, and Values.
- Multi-Head Attention: Parallel attention heads capture different relationships between tokens.
- Feed-Forward Networks: Per-token computation layers that process the attention output.
- Residual Connections: Skip connections improve gradient flow for deeper networks.
- Layer Normalization: Pre-norm implementation for stable training.
- Positional Embeddings: Add spatial information to the attention mechanism.

Technical Specifications
------------------------
- Framework: PyTorch
- Optimizer: AdamW
- Hardware Support: CPU and CUDA-enabled GPUs
- Regularization: Dropout to prevent overfitting

Training & Performance
----------------------
- Scaled-up model: 6 layers, 6 heads, 384 embedding dimensions
- Validation loss: 1.48 after about 60 minutes of training on a T4 GPU
- Generated text captures patterns and structure of Shakespearean English, though it may be nonsensical

