# RAG Pipeline (FastEmbed + pgvector)

This repository implements a Retrieval-Augmented Generation (RAG) pipeline over a local text corpus of scientific publications.

The storage layer is PostgreSQL with the pgvector extension, and text embeddings are generated locally using FastEmbed (ONNX).

## ✅ What this pipeline does

1. **Ingest text** ( from full papers and abstracts)
2. **Chunk + embed** locally → store embeddings in a Parquet file
3. **Load vectors into Postgres** (with IVFFlat index)
4. **Query** using pgvector
5. **Answer generation** (extractive or abstractive)

---

## Repository structure

```
.
├── full_paper_OCRed/                # Raw OCRed complete papers, each file is .txt
├── abstracts/                       # Abstracts, each file is .txt
├── scripts/
│   ├── 01_fast_embedding.py       # chunk + embed → writes bge_small.parquet
│   ├── 02_psql_upload.py          # bulk load parquet → PostgreSQL
│   ├── 03_naive_rag.py            # simple extractive retrieval-based QA
│   └── 04_MMR_abstractive_rag.py  # improved RAG with encoder-decoder generation and maximum relevance approach.
├── requirements.txt
└── README.md
```

> The knowledge based include the folders `full_paper_OCRed` and `abstracts`.

---

## Requirements

- Python 3.9+
- PostgreSQL 15+
- pgvector extension

macOS example:

```bash
brew install postgresql@16
brew services start postgresql@16
```

Create your venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Step 1 — Embed corpus to Parquet

```bash
python scripts/01_embed_texts_to_parquet.py     --input-dirs full_paper_OCRed abstracts     --out bge_small.parquet
```

---

## Step 2 — Load into PostgreSQL (vector store)

```bash
createdb ragdb
python scripts/02_load_parquet_to_pgvector.py     --parquet bge_small.parquet --dbname ragdb     --host localhost --port 5432 --user "$USER"
```

This builds:

- `passages` (vector column)
- `passage_blob` (original text)
- IVFFlat index

---

## Step 3 — Query

### Extractive naive answer

```bash
python scripts/03_query_pgvector_rag.py     --query "What is IVFFlat in pgvector?" --top-k 6     --dbname ragdb --host localhost --port 5432 --user "$USER"
```

### Abstractive (encoder-decoder) answer

```bash
python scripts/query_mmr_abstractive.py     --query "How do neurons and axons transmit signals?"     --top-k 8 --dbname ragdb
```

---

## Notes

- chunk overlap ~10–20% recommended
- `BAAI/bge-small-en-v1.5` is default embedding model
- IVFFlat `lists` ≈ N/1000 is a good heuristic
- If handling large Parquet (>100MB), do **not** commit to git — ignore or use Git LFS

---

## Future Enhancements

- cross-encoder reranker
- hybrid lexical + vector retrieval
- multi-database vector support

---

## Usage Examples

### Embed Knowledge Base

Generate embeddings and store them in a parquet file:

```bash
python scripts/01_embed_texts_to_parquet.py --input-dirs full_paper_OCRed abstracts --out bge_small.parquet
```

### Load Parquet into PostgreSQL vector store

```bash
python scripts/02_load_parquet_to_pgvector.py --parquet bge_small.parquet --dbname ragdb --host localhost --port 5432 --user "$USER"
```

### Ask a question (extractive)

```bash
python scripts/03_query_pgvector_rag.py --query "How do neurons fire?" --top-k 6 --dbname ragdb
```

### Ask a question (MMR + abstractive)

```bash
python scripts/query_mmr_abstractive.py --query "Explain neuronal action potentials" --top-k 8 --dbname ragdb
```
