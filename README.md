# 🧠 RAG Practice – Building a Retrieval-Augmented Generation Pipeline from Scratch

This project is my personal attempt to **understand how Retrieval-Augmented Generation (RAG)** works by implementing each component from the ground up.

The pipeline was built in Python, and the main goal was **learning by doing** — not just using high-level libraries like LangChain, but actually wiring up each stage myself.

---

## 🚀 What I Built

- ✅ Parsed and chunked Wikipedia-style text (from `wikitext-103`)
- ✅ Generated dense vector embeddings using `sentence-transformers`
- ✅ Indexed document chunks in FAISS for fast retrieval
- ✅ Retrieved relevant chunks based on a user question
- ✅ Used an LLM (e.g. Mistral-7B or FLAN-T5) to generate an answer from retrieved content
- ✅ Ran everything locally — including trying GPU inference

---

## 🎯 What I Wanted to Learn

- What RAG *really* is under the hood
- How to connect a retriever (FAISS) with a generator (transformers)
- How embeddings work and how to chunk documents effectively
- How GPU acceleration affects inference
- How to clean large files and secrets from Git history 😅

---

## 🛠 Stack

- Python
- Hugging Face Transformers
- SentenceTransformers
- FAISS
- Optional: bitsandbytes + CUDA
- Dataset: `wikitext-103` from Hugging Face 🤓

---

## 🧪 Current Interface

Everything runs from the terminal with:

```bash
python rag_app.py
