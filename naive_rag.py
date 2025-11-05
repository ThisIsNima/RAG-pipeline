import os
import argparse
import psycopg2
import textwrap
import re
from fastembed import TextEmbedding

def vec_literal(v):
    # pgvector text format '[v1,v2,...]'
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


def simple_generate_answer(question, rows, max_chars=900, max_sentences=5):
    """
    Naive version:
      - Iterate rows in ranked order.
      - From each row, take sentences in their original order.
      - Append the first sentences you see until you hit max_sentences or max_chars.
      - Add a simple [rank] citation after each sentence.
    """
    picked, used_cites, total = [], [], 0

    for rank_idx, (_pid, _score, _src, _cid, txt) in enumerate(rows, start=1):
        if not txt:
            continue
        # Very simple sentence split (period/!? + space). Not robust on abbreviations.
        sents = re.split(r"(?<=[.!?])\s+", txt.strip())
        for s in sents:
            s_clean = s.strip()
            if not s_clean:
                continue
            piece = f"{s_clean} [{rank_idx}]"
            # Enforce char cap, but allow at least one sentence overall
            if total + len(piece) > max_chars and picked:
                return " ".join(picked), sorted(set(used_cites))
            picked.append(piece)
            used_cites.append(rank_idx)
            total += len(piece)
            if len(picked) >= max_sentences:
                return " ".join(picked), sorted(set(used_cites))

    # Fallback: if nothing picked, show a shortened preview of the top row
    if not picked and rows:
        fallback = (rows[0][4] or "").strip().replace("\n", " ")
        picked = [textwrap.shorten(fallback, width=max_chars, placeholder=" …") + " [1]"]
        used_cites = [1]

    return " ".join(picked), sorted(set(used_cites))
# -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Query pgvector with a FastEmbed query embedding.")
    # Query args
    ap.add_argument("--query", "-q", required=True, help="Natural language question / query text")
    ap.add_argument("--top-k", "-k", type=int, default=5, help="Top K results")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="FastEmbed model name")
    ap.add_argument("--use-prefix", action="store_true",
                    help='Add "query: " prefix (recommended for BGE/E5)')
    # DB args
    ap.add_argument("--dbname", default="ragdb")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5432)
    ap.add_argument("--user", default=os.environ.get("USER"))
    ap.add_argument("--password", default=None)
    # Index tuning
    ap.add_argument("--ivf-probes", type=int, default=16, help="ivfflat.probes (ignored if HNSW)")
    ap.add_argument("--hnsw-ef", type=int, default=0, help="hnsw.ef_search (0 = unchanged)")
    args = ap.parse_args()

    # ---- Build query embedding----
    embedder = TextEmbedding(model_name=args.model)

    qtext = ("query: " + args.query) if args.use_prefix else args.query

    qvec = next(embedder.embed([qtext], normalize=True)).tolist()
    qvec_lit = vec_literal(qvec)

    # ---- Connect to Postgres ----
    conn = psycopg2.connect(
        dbname=args.dbname, host=args.host, port=args.port,
        user=args.user, password=args.password
    )
    cur = conn.cursor()

 
    cur.execute("SET LOCAL ivfflat.probes = %s;", (max(1, args.ivf_probes),))

    if args.hnsw_ef > 0:
        cur.execute("SET LOCAL hnsw.ef_search = %s;", (args.hnsw_ef,))

    # ---- Search based on cosine distance----
    sql = """
    WITH q AS (SELECT %s::vector AS v)
    SELECT p.id,
           1 - (p.embedding <=> q.v) AS score,   -- cosine similarity in [0,1]
           b.source_path, b.chunk_id,
           b.text
    FROM passages p
    JOIN passage_blob b USING (id), q
    ORDER BY p.embedding <=> q.v
    LIMIT %s;
    """
    cur.execute(sql, (qvec_lit, args.top_k))
    rows = cur.fetchall()
    cur.close(); conn.close()

    # ---- print retrieval ----
    for rank, (pid, score, src, cid, txt) in enumerate(rows, 1):
        preview = textwrap.shorten((txt or "").replace("\n", " "), width=400, placeholder=" …")
        print(f"[{rank}] id={pid}  score={score:.3f}")
        print(f"     {src}  (chunk {cid})")
        print(f"     {preview}\n")

    # ---- ----
    answer, cites = simple_generate_answer(args.query, rows)
    print("=== Answer ===")
    print(answer)
    print("\n=== Sources ===")
    for idx in cites:
        pid, score, src, cid, _ = rows[idx-1]
        print(f"[{idx}] score={float(score):.3f}  {src} (chunk {int(cid)})")

if __name__ == "__main__":
    main()
