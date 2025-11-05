#!/usr/bin/env python3
import os, argparse, psycopg2, textwrap, re
from fastembed import TextEmbedding

def vec_literal(v):
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"

# --------- NEW: one method to evaluate retrieval & generation P/R ----------
def evaluate_pr(rows, answer_text=None, gold_ids=None, gold_sources=None, gold_answer=None):
    """
    rows: list of (id, score, source_path, chunk_id, text)
    answer_text: generated (or extracted) answer string
    gold_ids: set[int] of DB ids considered relevant (passages.id)
    gold_sources: set[str] of substrings; if a row's source_path contains any, it's relevant
    gold_answer: reference answer string for generation P/R

    Returns dict with keys (present if corresponding gold provided):
      - Retrieval: 'retrieval_precision', 'retrieval_recall', 'retrieval_tp', 'retrieval_k', 'retrieval_gold'
      - Generation (BERTScore): 'gen_precision', 'gen_recall', 'gen_f1', 'gen_metric'
        (Falls back to token-overlap if bert-score unavailable.)
    """
    import re
    metrics = {}

    # ---- Retrieval P/R (unchanged) ----
    if gold_ids or gold_sources:
        retrieved_ids = {int(r[0]) for r in rows}
        relevant = set()

        if gold_ids:
            relevant |= (retrieved_ids & set(map(int, gold_ids)))

        if gold_sources:
            src_set = set()
            for (pid, _score, src, _cid, _txt) in rows:
                if any(gs.lower() in (src or "").lower() for gs in gold_sources):
                    src_set.add(pid)
            relevant |= src_set

        tp = len(relevant)
        k = len(rows)
        gold_count = (len(gold_ids) if gold_ids else 0)
        if gold_sources and not gold_ids:
            gold_count = len(gold_sources)

        prec = tp / k if k else 0.0
        rec  = tp / gold_count if gold_count else 0.0

        metrics.update({
            "retrieval_precision": round(prec, 4),
            "retrieval_recall": round(rec, 4),
            "retrieval_tp": tp,
            "retrieval_k": k,
            "retrieval_gold": gold_count,
        })

    # ---- Generation P/R via BERTScore (with fallback) ----
    if gold_answer is not None and answer_text is not None:
        try:
            # Lazy import to avoid extra deps unless needed
            from bert_score import score as bertscore_score

            # Allow overriding model/baseline via env vars (optional):
            #   BERTSCORE_MODEL=roberta-large (default)
            #   BERTSCORE_RESCALE=1 to enable baseline rescaling
            model_type = os.environ.get("BERTSCORE_MODEL", "roberta-large")
            rescale    = os.environ.get("BERTSCORE_RESCALE", "0") == "1"

            P, R, F1 = bertscore_score(
                [answer_text], [gold_answer],
                lang="en",
                model_type=model_type,
                rescale_with_baseline=rescale
            )
            p = float(P.mean()); r = float(R.mean()); f1 = float(F1.mean())
            metrics.update({
                "gen_precision": round(p, 4),
                "gen_recall":    round(r, 4),
                "gen_f1":        round(f1, 4),
                "gen_metric":    f"bertscore({model_type}{',rescaled' if rescale else ''})",
            })
        except Exception:
            # Fallback: token-overlap precision/recall
            tok = lambda s: set(re.findall(r"[A-Za-z0-9]+", s.lower()))
            pred_tokens = tok(answer_text)
            gold_tokens = tok(gold_answer)
            inter = pred_tokens & gold_tokens
            p = len(inter) / len(pred_tokens) if pred_tokens else 0.0
            r = len(inter) / len(gold_tokens) if gold_tokens else 0.0
            f1 = (2*p*r)/(p+r) if (p+r) > 0 else 0.0
            metrics.update({
                "gen_precision": round(p, 4),
                "gen_recall":    round(r, 4),
                "gen_f1":        round(f1, 4),
                "gen_metric":    "token-overlap(fallback)",
            })

    return metrics



def mmr_generate_answer(
    question,
    rows,
    embed_model="BAAI/bge-small-en-v1.5",
    max_sentences=6,
    max_chars=1200,
    alpha=0.7,
    gen_model="google/flan-t5-base",
    max_new_tokens=180,
    num_beams=4,
    temperature=0.7,
    top_p=0.95,
    device=None,  # e.g., "cuda" or "cpu"; if None, auto-detect
):
    """
    Build an abstractive answer using:
      1) Sentence-level MMR selection (FastEmbed embeddings)
      2) Encoder-decoder generation (e.g., FLAN-T5) over the selected snippets

    rows: [(id, score, source_path, chunk_id, text), ...]
    Returns (answer_text, cites_used_sorted)
    """
    import re, numpy as np, torch
    from fastembed import TextEmbedding
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    # ---------------------------
    # 0) Device
    # ---------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------
    # 1) Sentence candidates with metadata
    # ---------------------------
    cands = []  # [(sent, src, cid, rank_idx)]
    for rank_idx, (_pid, _score, src, cid, txt) in enumerate(rows, start=1):
        if not txt:
            continue
        sents = re.split(r"(?<=[.!?])\s+", (txt or "").strip())
        for s in sents:
            s = s.strip()
            if 40 <= len(s) <= 400:  # filter very short/long
                cands.append((s, src or "", int(cid), rank_idx))

    if not cands:
        # fallback: take a short preview from the top row (still abstractive)
        preview = (rows[0][4] or "").strip().replace("\n", " ") if rows else ""
        context_preview = preview[:max_chars]
        # minimal generation on preview
        tok = AutoTokenizer.from_pretrained(gen_model)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(gen_model).to(device)
        prompt = (
            "Answer the question using only the given context.\n\n"
            f"Question: {question}\n\nContext:\n{context_preview}\n\n"
            "Answer:"
        )
        inputs = tok(prompt, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            out = mdl.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=(temperature is not None and temperature > 0),
                temperature=temperature if temperature else 1.0,
                top_p=top_p,
            )
        ans = tok.decode(out[0], skip_special_tokens=True).strip()
        cites = [1] if rows else []
        return ans, cites

    # ---------------------------
    # 2) Embed query + sentences (normalized) with FastEmbed (CPU)
    # ---------------------------
    emb = TextEmbedding(model_name=embed_model)
    qvec = next(emb.embed(["query: " + question], normalize=True))
    sent_texts = [s for s, _, _, _ in cands]
    sent_vecs = list(emb.embed(sent_texts, normalize=True))
    X = np.vstack(sent_vecs).astype("float32")   # (m, d)
    qv = np.asarray(qvec, dtype="float32")       # (d,)

    # ---------------------------
    # 3) Similarity to query (cosine since normalized)
    # ---------------------------
    sims = X @ qv

    # ---------------------------
    # 4) MMR selection (diverse, relevant)
    # ---------------------------
    selected = []
    selected_idx = []
    max_sims_to_sel = np.zeros_like(sims) - 1.0

    def sent_sim(i, j):
        return float(np.dot(X[i], X[j]))

    for _ in range(min(max_sentences, len(cands))):
        if not selected_idx:
            i = int(np.argmax(sims))
            selected_idx.append(i); selected.append(cands[i])
            continue
        # update redundancy scores against current selection
        for i in range(len(cands)):
            if i in selected_idx:
                continue
            max_sims_to_sel[i] = max(max_sims_to_sel[i], max(sent_sim(i, j) for j in selected_idx))
        # classic MMR: alpha * relevance - (1-alpha) * redundancy
        # equivalently: alpha*sims + (1-alpha)*(1 - max_sim_to_sel)
        mmr = alpha * sims + (1 - alpha) * (1 - max_sims_to_sel)
        mmr[selected_idx] = -1e9
        i = int(np.argmax(mmr))
        if mmr[i] < 0.05:  # nothing useful left
            break
        selected_idx.append(i); selected.append(cands[i])

    # ---------------------------
    # 5) Order selected sentences to form a coherent context
    # ---------------------------
    selected_ordered = sorted(selected, key=lambda t: (t[1], t[2], t[3]))

    # ---------------------------
    # 6) Build a concise context budget with citations
    # ---------------------------
    context_lines, cites_set, total = [], set(), 0
    for sent, src, cid, rank_idx in selected_ordered:
        piece = sent
        if total + len(piece) > max_chars and context_lines:
            break
        # Keep a simple bullet format to help T5 models
        context_lines.append(f"- [{rank_idx}] {piece}")
        cites_set.add(rank_idx)
        total += len(piece)

    # Fallback if nothing fit the budget for some reason
    if not context_lines:
        sent, src, cid, rank_idx = selected_ordered[0]
        context_lines = [f"- [{rank_idx}] {sent[:max_chars]}"]
        cites_set = {rank_idx}

    # ---------------------------
    # 7) Abstractive generation with an encoder-decoder model
    # ---------------------------
    context_text = "\n".join(context_lines)

    # A compact, instruction-tuned prompt that tends to work well with FLAN-T5/BART
    prompt = (
        "Answer the question using ONLY the context snippets. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Question:\n{question}\n\n"
        f"Context snippets:\n{context_text}\n\n"
        "Answer:"
    )

    tok = AutoTokenizer.from_pretrained(gen_model)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(gen_model).to(device)

    inputs = tok(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        output_ids = mdl.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=(temperature is not None and temperature > 0),
            temperature=temperature if temperature else 1.0,
            top_p=top_p,
        )
    answer = tok.decode(output_ids[0], skip_special_tokens=True).strip()

    # You can choose to append a compact citation footer like “[Sources: 1, 3, 7]”
    cites_sorted = sorted(cites_set)
    return answer, cites_sorted




# --------------------------------------------------------------------------

# -------- your simple generator from before (unchanged) --------
def simple_generate_answer(question, rows, max_chars=900, max_sentences=5):
    q_terms = set(re.findall(r"[A-Za-z0-9]{3,}", question.lower()))
    if not q_terms: q_terms = set(question.lower().split())
    candidates = []
    for rank_idx, (_pid, score, _src, _cid, txt) in enumerate(rows, start=1):
        if not txt: continue
        sents = re.split(r"(?<=[.!?])\s+", txt.strip())
        for s in sents:
            s_clean = s.strip()
            if not s_clean: continue
            words = set(re.findall(r"[A-Za-z0-9]{3,}", s_clean.lower()))
            if not words: continue
            overlap = len(q_terms & words)
            candidates.append((overlap + float(score)*0.5, s_clean, rank_idx))

    candidates.sort(key=lambda x: x[0], reverse=True)
    picked, used_cites, total = [], [], 0
    seen = set()
    for sc, sent, cite in candidates:
        key = (sent[:80].lower(), cite)
        if key in seen: continue
        seen.add(key)
        piece = f"{sent} [{cite}]"
        if total + len(piece) > max_chars and picked: break
        picked.append(piece); used_cites.append(cite); total += len(piece)
        if len(picked) >= max_sentences: break

    if not picked:
        fallback = (rows[0][4] or "").strip().replace("\n", " ")
        picked = [textwrap.shorten(fallback, width=max_chars, placeholder=" …") + " [1]"]
        used_cites = [1]

    return " ".join(picked), sorted(set(used_cites))
# ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Query pgvector with a FastEmbed query embedding.")
    ap.add_argument("--query", "-q", required=True)
    ap.add_argument("--top-k", "-k", type=int, default=5)
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--use-prefix", action="store_true")
    # DB
    ap.add_argument("--dbname", default="ragdb")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5432)
    ap.add_argument("--user", default=os.environ.get("USER"))
    ap.add_argument("--password", default=None)
    # Index tuning
    ap.add_argument("--ivf-probes", type=int, default=16)
    ap.add_argument("--hnsw-ef", type=int, default=0)
    # --- Eval inputs (optional) ---
    ap.add_argument("--gold-ids", type=str, default=None,
                    help="Comma-separated passage IDs that are relevant (e.g. 12,45,90)")
    ap.add_argument("--gold-sources", type=str, default=None,
                    help="Comma-separated substrings to match in source_path as relevant")
    ap.add_argument("--gold-answer", type=str, default=None,
                    help="Gold reference answer text")
    ap.add_argument("--gold-answer-file", type=str, default=None,
                    help="Path to a file containing the gold reference answer")
    args = ap.parse_args()

    # Build query embedding
    embedder = TextEmbedding(model_name=args.model)
    qtext = ("query: " + args.query) if args.use_prefix else args.query
    qvec = next(embedder.embed([qtext], normalize=True)).tolist()
    qvec_lit = vec_literal(qvec)

    # Connect + search
    conn = psycopg2.connect(dbname=args.dbname, host=args.host, port=args.port,
                            user=args.user, password=args.password)
    cur = conn.cursor()
    cur.execute("SET LOCAL ivfflat.probes = %s;", (max(1, args.ivf_probes),))
    if args.hnsw_ef > 0:
        cur.execute("SET LOCAL hnsw.ef_search = %s;", (args.hnsw_ef,))
    sql = """
    WITH q AS (SELECT %s::vector AS v)
    SELECT p.id, 1 - (p.embedding <=> q.v) AS score, b.source_path, b.chunk_id, b.text
    FROM passages p JOIN passage_blob b USING (id), q
    ORDER BY p.embedding <=> q.v
    LIMIT %s;
    """
    cur.execute(sql, (qvec_lit, args.top_k))
    rows = cur.fetchall()
    cur.close(); conn.close()

    # Show retrieved
    for rank, (pid, score, src, cid, txt) in enumerate(rows, 1):
        preview = textwrap.shorten((txt or "").replace("\n", " "), width=400, placeholder=" …")
        print(f"[{rank}] id={pid}  score={score:.3f}")
        print(f"     {src}  (chunk {cid})")
        print(f"     {preview}\n")




        # ---- 4) Print top-k nicely ----
    for rank, (pid, score, src, cid, txt) in enumerate(rows, 1):
        preview = textwrap.shorten((txt or "").replace("\n", " "), width=400, placeholder=" …")
        print(f"[{rank}] id={pid}  score={score:.3f}")
        print(f"     {src}  (chunk {cid})")
        print(f"     {preview}\n")

    # ---- 5) MMR-based generator ----
    if rows:
        answer, cites = mmr_generate_answer(args.query, rows,
                                            embed_model=args.model,  # keep same family
                                            max_sentences=6, max_chars=1200, alpha=0.7)
        print("=== Answer ===")
        print(answer)
        print("\n=== Sources ===")
        for idx in cites:
            pid, score, src, cid, _ = rows[idx-1]
            print(f"[{idx}] score={float(score):.3f}  {src} (chunk {int(cid)})")


if __name__ == "__main__":
    main()
