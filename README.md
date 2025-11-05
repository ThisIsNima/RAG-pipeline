# RAG-pipeline
A RAG pipeline that uses a collection of scientific publications as the knowledge base
# RAG-on-Text: FastEmbed + pgvector Starter

End-to-end recipe to:

1) chunk & embed raw text files from a local knowledge base  
2) store vectors in PostgreSQL/pgvector with an IVFFlat index  
3) run a very naive Q&A retriever that prints sources with inline citations

This repo expects two text-corpus folders:

```
full_paper_OCRed/   # OCRed full papers, one .txt per document
abstracts/          # abstracts, one .txt per document
```

> You can add more folders later; the embedding script accepts multiple input directories.

---

## Repository layout

```
.
├── full_paper_OCRed/              # <- your KB (text files)
├── abstracts/                     # <- your KB (text files)
├── scripts/
│   ├── 01_embed_texts_to_parquet.py      # create chunks + embeddings (FastEmbed / ONNX)
│   ├── 02_load_parquet_to_pgvector.py    # bulk-load embeddings to Postgres
│   ├── 03_query_pgvector_rag.py          # naive retrieval + extractive answer with citations
│   └── 03_query_with_eval.py             # retrieval + optional P/R evaluation
├── requirements.txt
└── README.md     # (this file)
```

---

## Prerequisites

- Python 3.9+
- PostgreSQL 15+ with pgvector extension installed
- macOS:  
  ```
  brew install postgresql@16
  brew services start postgresql@16
  ```
- virtualenv recommended:
  ```
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  ```

---

## install python packages

```
pip install -r requirements.txt
```

---

## 1) build embeddings parquet from your KB

this will walk folders, chunk text, embed with fastembed, and write a parquet:

```
python scripts/01_embed_texts_to_parquet.py   --input-dirs full_paper_OCRed abstracts   --out bge_small.parquet
```

---

## 2) create tables & bulk-load embeddings into pgvector

make db:

```
createdb ragdb
```

load:

```
python scripts/02_load_parquet_to_pgvector.py   --parquet bge_small.parquet   --dbname ragdb --host localhost --port 5432 --user "$USER"
```

this script:

- CREATE EXTENSION vector
- makes 2 tables: `passages` + `passage_blob`
- bulk inserts all embeddings
- creates IVFFlat index with cosine ops

---

## 3) ask questions (naive extractive answer)

```
python scripts/03_query_pgvector_rag.py   --query "How do neurons and axons transmit signals?"   --top-k 6   --dbname ragdb --host localhost --port 5432 --user "$USER"
```

will print:

- top-K passages
- naive stitched answer
- numbered citations mapping back to file + chunk

---

## optional: P/R evaluation hooks

```
python scripts/03_query_with_eval.py   --query "What is IVFFlat and how does retrieval work?"   --top-k 8   --dbname ragdb --host localhost --port 5432 --user "$USER"
```

this version has optional arguments for gold sets (ids or substrings) and reference answer. it can compute retrieval precision/recall signals.

---

## data model

- `passages` (vector store)
- `passage_blob` (original text + metadata)

the embedding column is pgvector.

index: IVFFlat + cosine ops.

---

## tuning notes

- chunk overlap can help a lot: 10-25% is common
- for IVFFlat lists: start at N/1000
- cosine similarity expects normalized vectors

---

## extending

- swap another FastEmbed model
- add cross-encoder reranker
- replace naive sentence picker with LLM generator

---

## license

MIT (or specify)

---
