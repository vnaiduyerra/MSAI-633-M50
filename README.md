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

## Project layout

```
app.py                 # RAG pipeline + Gradio UI
data/cricket_rules.md  # corpus
requirements.txt
```
