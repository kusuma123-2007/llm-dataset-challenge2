

---

# **LLM Micro Challenge — Challenge 2**
Custom Tokenizer Creation (Health Domain)

This repository contains the deliverables for Challenge 2 of the LLM Micro Challenge Series, where the objective was to build a custom tokenizer trained specifically on the Health domain dataset created in Challenge 1.

# Why Build a Custom Tokenizer?

A domain-specific tokenizer improves the model’s ability to understand and process health-related terminology.
Benefits include:

Better handling of complex medical terms

Fewer unknown tokens

More efficient vocabulary usage

Improved downstream model performance

Examples of health-related subwords learned:

cardio + vascular

hyper + tension

immuno + therapy

neuro + logical

respiratory + system

## **Challenge Breakdown**
1. Tokenizer Type: Byte Pair Encoding (BPE)

BPE was selected for this challenge because:

It handles rare medical words efficiently

Reduces vocabulary size

Works well with domain-specific datasets

Is widely used in modern large language models

Special tokens included:

[PAD], [UNK], [CLS], [SEP]

# **2. Training Data**

The tokenizer was trained exclusively on the file:

health_dataset_clean.jsonl


This dataset contains a wide variety of health content, covering:

Anatomy and physiology

Diseases and symptoms

Nutrition and wellness

First aid and public health

Medical terminology

## **3. Tokenizer Training Process**

A custom script named train_tokenizer.py was used to:

Load all JSONL entries

Normalize text

Apply whitespace pre-tokenization

Train a BPE model

Generate vocabulary and merge rules

Save the tokenizer in Hugging Face format

Training produced the following output files:

vocab.json

merges.txt

tokenizer.json

These files are compatible with common LLM training pipelines.

## **4. Tokenizer Evaluation**

# Example test:

Input:

Dehydration can cause dizziness, fatigue, and headache.


# Output Tokens:

['▁dehydration', '▁can', '▁cause', '▁diz', 'ziness', '▁fatigue', '▁and', '▁head', 'ache', '.']


This demonstrates correct segmentation of medical terms and consistent BPE behavior.

# **Repository Structure**
File Name	Description
health_dataset_clean.jsonl	Dataset used for tokenizer training
train_tokenizer.py	Script to train the tokenizer
tokenizer_out/vocab.json	Token vocabulary file
tokenizer_out/merges.txt	Merge rules learned by BPE
tokenizer_out/tokenizer.json	Full tokenizer model configuration
sample_tests.txt	Test sentences for validation

## **Outcome : **

This challenge produced a fully functional, domain-optimized tokenizer tailored for the Health dataset.
It is now ready for:

Pretraining
Fine-tuning
Embedding generation
health-focused AI applications
Chatbots and Q&A systems
