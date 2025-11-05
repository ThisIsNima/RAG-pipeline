import os, sys, time, re
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# ---------- Config ----------
PARQUET = "bge_small.parquet"   
DBNAME  = "ragdb"
HOST    = "localhost"
PORT    = 5432
USER    = os.environ.get("USER")  

BATCH_ROWS = 5000    
PAGE_SIZE  = 1000    
IVF_LISTS  = 128     
MAINT_WORK_MEM = "512MB" 



CTRL_RE = re.compile(r'[\x00-\x08\x0B-\x1F]')

def clean_text(s) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return CTRL_RE.sub(" ", s)

def vec_to_pg(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in v.tolist()) + "]"

def main():

    df = pd.read_parquet(PARQUET)
    if "embedding" not in df.columns:
        print("No 'embedding' column in parquet.", file=sys.stderr); sys.exit(2)
    dim = len(df["embedding"].iloc[0])
    total = len(df)
    print(f"Loaded {total} rows from {PARQUET} (dim={dim})")

    rows_vec, rows_blob = [], []
    t0 = time.time()
    for _, r in df.iterrows():
        v = np.asarray(r["embedding"], dtype=np.float32)
        v /= (np.linalg.norm(v) + 1e-12)  
        rows_vec.append((vec_to_pg(v),))
        rows_blob.append((
            clean_text(r.get("text", "")),
            clean_text(r.get("source_path", "")),
            int(r.get("chunk_id", 0)),
        ))
    print(f"Prepared rows in {time.time()-t0:.2f}s")
 #-----connect DDL and database---------------

    conn = psycopg2.connect(dbname=DBNAME, host=HOST, port=PORT, user=USER)
    cur  = conn.cursor()

    conn.autocommit = True 
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS passages (
          id BIGSERIAL PRIMARY KEY,
          embedding vector({dim})
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS passage_blob (
          id bigint PRIMARY KEY REFERENCES passages(id),
          text text,
          source_path text,
          chunk_id int
        );
    """)
 #-------batch insert -----------------------

    inserted = 0
    conn.autocommit = False 
    try:
        for start in range(0, total, BATCH_ROWS):
            end = min(start + BATCH_ROWS, total)
            chunk_vecs  = rows_vec[start:end]
            chunk_blobs = rows_blob[start:end]

            ids_rows = execute_values(
                cur,
                "INSERT INTO passages (embedding) VALUES %s RETURNING id;",
                chunk_vecs,
                template="(%s::vector)",
                page_size=PAGE_SIZE,
                fetch=True,  # collect ALL ids from paged inserts
            )
            ids = [r[0] for r in ids_rows]
            assert len(ids) == len(chunk_blobs)

            execute_values(
                cur,
                "INSERT INTO passage_blob (id, text, source_path, chunk_id) VALUES %s;",
                [(i, t, s, c) for i, (t, s, c) in zip(ids, chunk_blobs)],
                page_size=PAGE_SIZE,
            )

            conn.commit()
            inserted += (end - start)
            print(f"  Batch {start}-{end} committed ({inserted}/{total})")
    except Exception as e:
        conn.rollback()
        print(f"ERROR during inserts, rolled back current batch: {e}", file=sys.stderr)
        raise
    finally:
        conn.autocommit = True


    cur.execute(f"SET maintenance_work_mem = '{MAINT_WORK_MEM}';")
    cur.execute("DROP INDEX IF EXISTS passages_embedding_ivf;")
    cur.execute(f"""
        CREATE INDEX passages_embedding_ivf
        ON passages USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = {IVF_LISTS});
    """)

    cur.close(); conn.close()

if __name__ == "__main__":
    main()
