---
title: Cricket Rules RAG Chatbot
emoji: 🏏
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
---

# MSAI-633-M50 — Cricket Rules RAG Chatbot

A Retrieval-Augmented Generation chatbot that answers questions about the Laws of
Cricket using only `data/cricket_rules.md`. Gradio frontend, Hugging Face models:

| Stage      | Model / method                                   |
|------------|--------------------------------------------------|
| Embedding  | `sentence-transformers/all-MiniLM-L6-v2` + FAISS |
| Retrieval  | Dense + BM25, fused with Reciprocal Rank Fusion  |
| Reranking  | `cross-encoder/ms-marco-MiniLM-L-6-v2`           |
| Generation | `google/flan-t5-small`                           |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

The first run downloads the models from Hugging Face (~400 MB). Then open
http://localhost:7861. A public `gradio.live` link is also printed; set
`GRADIO_SHARE=0` to turn it off, or `PORT=xxxx` to change the port.

## Deploy to a Hugging Face Space

This README's YAML header is the Space config (Gradio SDK, `app.py`). Push the repo
to your Space:

```bash
git remote add space https://huggingface.co/spaces/<your-username>/<your-space>
git push space HEAD:main        # password = a Hugging Face write token
```

The Space installs `requirements.txt`, downloads the models on first boot (a few
minutes on the free CPU tier), then serves the chatbot.

## Project layout

```
app.py                 # RAG pipeline + Gradio UI
data/cricket_rules.md  # corpus
requirements.txt
```
