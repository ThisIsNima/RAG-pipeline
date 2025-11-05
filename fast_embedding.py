#!/usr/bin/env python3
import os
# keep things single-threaded & quiet on CPU
os.environ["ONNX_CPU_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse, json, time
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np

from fastembed import TextEmbedding  # CPU-only by default

def approx_char_window(tokens:int) -> int:
    return max(256, int(tokens * 4))  # ~4 chars/token heuristic

def char_chunks(text:str, chunk_tokens:int, overlap_tokens:int):
    if not text or not text.strip():
        return []
    win = approx_char_window(chunk_tokens)
    step = max(1, win - approx_char_window(overlap_tokens))
    chunks = []
    n = len(text)
    for start in range(0, n, step):
        end = min(start + win, n)
        ch = text[start:end].strip()
        if ch:
            chunks.append((start, end, ch))
        if end >= n:
            break
    return chunks

def main():
    ap = argparse.ArgumentParser(description="CPU-only embedding with FastEmbed (no HF tokenizers).")
    ap.add_argument("input_dir", help="Folder with .txt files (recurses)")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5",
                    help="FastEmbed model (e.g., 'BAAI/bge-small-en-v1.5', 'paraphrase-MiniLM-L6-v2')")
    ap.add_argument("--chunk-tokens", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--doc-prefix", default="", help='e.g., "passage: " for BGE/E5-style docs')
    ap.add_argument("--out", default="embeddings.parquet")
    ap.add_argument("--also-csv", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.input_dir).expanduser().resolve()
    files = sorted([p for p in in_dir.rglob("*.txt") if p.is_file()])
    if not files:
        print(f"No .txt files under: {in_dir}")
        return

    print(f"Loading FastEmbed model: {args.model}")
    embedder = TextEmbedding(model_name=args.model)

    rows = []
    total_chunks = 0
    t0 = time.time()

    for i, f in enumerate(files, 1):
        try:
            txt = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            txt = f.read_text(encoding="latin-1", errors="ignore")

        chunks = char_chunks(txt, args.chunk_tokens, args.overlap)
        if not chunks:
            chunks = [(0, min(2000, len(txt)), txt[:2000])]

        docs = [args.doc_prefix + c[2] for c in chunks]

        t_start = time.time()
        # FastEmbed returns a generator of numpy arrays
        vecs = list(embedder.embed(docs, batch_size=args.batch_size, normalize=args.normalize))
        dt = time.time() - t_start

        for j, ((start_c, end_c, chunk_text), v) in enumerate(zip(chunks, vecs)):
            rows.append({
                "source_path": str(f),
                "chunk_id": j,
                "start_char": start_c,
                "end_char": end_c,
                "text": chunk_text,
                "embedding": v.astype(np.float32).tolist(),
            })

        total_chunks += len(chunks)
        avg_ms = (dt / max(1, len(chunks))) * 1000
        print(f"[{i}/{len(files)}] {f.name}: {len(chunks)} chunks in {dt:.2f}s "
              f"(avg {avg_ms:.1f} ms/chunk) | total_chunks={total_chunks}")

    dim = len(rows[0]["embedding"]) if rows else 0
    df = pd.DataFrame(rows)
    df.attrs.update({
        "model": args.model,
        "embedding_dim": dim,
        "chunk_tokens_approx": args.chunk_tokens,
        "overlap_tokens_approx": args.overlap,
        "doc_prefix": args.doc_prefix,
    })

    out_path = Path(args.out).resolve()
    print(f"Writing {len(df)} chunks (dim={dim}) to {out_path}")
    df.to_parquet(out_path, index=False)

    if args.also_csv:
        csv_path = out_path.with_suffix(".csv")
        print(f"Also writing CSV to {csv_path} (warning: large)")
        df_csv = df.copy()
        df_csv["embedding"] = df_csv["embedding"].apply(lambda v: json.dumps(v))
        df_csv.to_csv(csv_path, index=False)

    print(f"Done in {time.time() - t0:.2f}s.")

if __name__ == "__main__":
    main()
