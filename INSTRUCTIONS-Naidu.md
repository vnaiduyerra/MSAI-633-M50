# Naidu (Retrieval lead) — part 3 of 5

These are your files for the team repository. Read them before you commit:
your name on the commit means you have reviewed this code and can explain it.

## What you should be able to explain
- Two chunking strategies: fixed (512 tokens, 64 overlap) and structure-aware (headings, max 800 tokens, each table its own chunk). Chunk ID = doc_id:strategy-version:sequence. PDFs are split by page.
- Embeddings: BAAI/bge-small-en-v1.5 (384 dims) in embedded Qdrant, cosine similarity, about 12,500 chunks. Re-chunking writes a new version, checks coverage, switches, then deletes the old version.
- Retrieval: dense + BM25 fused with reciprocal rank fusion (30 candidates), bge-reranker-base keeps the top 5. Superseded editions are skipped unless the question names a year; near-duplicate clauses are collapsed.

## Files in this zip
- cricketrag/chunk/__init__.py
- cricketrag/chunk/chunker.py
- cricketrag/index/__init__.py
- cricketrag/index/build.py
- cricketrag/retrieve/__init__.py
- cricketrag/retrieve/hybrid.py

## Steps
1. Wait until part 2 has been merged into main, because your code imports it.
2. Get the repository (skip the clone if you already have it):

       git clone https://github.com/srikanthj8508/cricket-rules-rag-chatbot.git
       cd cricket-rules-rag-chatbot
       git checkout main && git pull origin main

3. Create your branch:

       git checkout -b feature/chunk-index-retrieve

4. Unzip this file and copy its contents into the repository folder, keeping the
   folder structure. Do not copy this instruction file.
5. Commit and push under your own GitHub account:

       git add -A
       git status          # check the list matches "Files in this zip"
       git commit -m "Add chunking, index build and hybrid retrieval"
       git push -u origin feature/chunk-index-retrieve

6. Open the pull request link that GitHub prints. Ask **Naveen** to review and approve it, then merge.


## Note on authorship
This code was drafted with AI assistance (Claude); the README says so. Add your
own improvements, fixes or report sections as further commits on your branch.
