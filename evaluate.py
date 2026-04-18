# eval/evaluate.py - runs evaluation metrics for the RFP search engine
#
# Usage:
#   python eval/evaluate.py
#   python eval/evaluate.py --generate-candidates
#   python eval/evaluate.py --data rfp.csv --queries eval/queries.json

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# add project root to path so we can import main
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import main as search_engine

# file paths
QUERIES_FILE   = ROOT / "eval" / "queries.json"
RESULTS_DIR    = ROOT / "eval" / "results"
CANDIDATES_CSV = ROOT / "eval" / "candidates.csv"
METRICS_CSV    = RESULTS_DIR / "metrics.csv"

CONFIGS = [
    {"name": "tfidf",      "mode": "tfidf",   "alpha": 0.5},
    {"name": "sbert",      "mode": "semantic", "alpha": 0.5},
    {"name": "hybrid-0.3", "mode": "hybrid",   "alpha": 0.3},
    {"name": "hybrid-0.5", "mode": "hybrid",   "alpha": 0.5},
    {"name": "hybrid-0.7", "mode": "hybrid",   "alpha": 0.7},
]

K = 10
CANDIDATE_K = 30
CORPUS_SIZE = 500  # used for full-corpus MAP computation


# metric functions

def precision_at_k(ranked_ids, relevant, k):
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant)
    return hits / k


def recall_at_k(ranked_ids, relevant, k):
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant)
    return hits / len(relevant)


def average_precision(ranked_ids, relevant):
    """AP over the full ranked list."""
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            hits += 1
            score += hits / rank
    return score / len(relevant)


def dcg_at_k(ranked_ids, relevant, k):
    score = 0.0
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in relevant:
            score += 1.0 / np.log2(rank + 1)
    return score


def ndcg_at_k(ranked_ids, relevant, k):
    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1.0 / np.log2(r + 2) for r in range(ideal_hits))
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(ranked_ids, relevant, k) / ideal_dcg


def run_config(query, cfg, top_k):
    result_df = search_engine.run_search(
        query, cfg["mode"], top_k=top_k, alpha=cfg["alpha"]
    )
    return result_df["id"].tolist()


# candidate generation for manual labeling

def generate_candidates(queries):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # load any existing labels so we don't overwrite them
    existing_labels = {}
    if CANDIDATES_CSV.exists():
        with open(CANDIDATES_CSV, encoding="utf-8") as f:
            for rec in csv.DictReader(f):
                key = (rec["query_id"], rec["doc_id"])
                if rec.get("relevant", "").strip() != "":
                    existing_labels[key] = rec["relevant"].strip()

    rows = []
    carried_over = 0
    new_unlabeled = 0

    for q in queries:
        seen = {}
        methods_seen = {}

        for cfg in CONFIGS:
            result_df = search_engine.run_search(
                q["query"], cfg["mode"], top_k=CANDIDATE_K, alpha=cfg["alpha"]
            )
            for _, row in result_df.iterrows():
                doc_id = row.get("id", "")
                if not doc_id:
                    continue
                methods_seen.setdefault(doc_id, []).append(cfg["name"])
                if doc_id not in seen:
                    seen[doc_id] = row

        # sort by hybrid-0.5 score
        hybrid_results = search_engine.run_search(
            q["query"], "hybrid", top_k=CANDIDATE_K, alpha=0.5
        )
        hybrid_order = {row["id"]: i for i, row in hybrid_results.iterrows()}

        sorted_docs = sorted(
            seen.keys(),
            key=lambda d: hybrid_order.get(d, 99999)
        )

        for rank, doc_id in enumerate(sorted_docs, start=1):
            row = seen[doc_id]
            desc = (row.get("description_text") or row.get("combined_text") or "").strip()
            desc = desc.replace("\n", " ")
            snippet = desc[:300] if desc else ""

            url = ""
            for col in ("source_url", "ui_link", "additional_info_link"):
                val = row.get(col)
                if isinstance(val, str) and val.startswith("http"):
                    url = val
                    break

            key = (q["id"], doc_id)
            existing = existing_labels.get(key, "")
            if existing != "":
                carried_over += 1
            else:
                new_unlabeled += 1

            rows.append({
                "query_id":   q["id"],
                "query":      q["query"],
                "category":   q["category"],
                "rank":       rank,
                "doc_id":     doc_id,
                "title":      (row.get("title") or "")[:120],
                "snippet":    snippet,
                "url":        url,
                "score":      f"{float(row.get('score', 0)):.4f}",
                "method":     "|".join(methods_seen[doc_id]),
                "relevant":   existing,
            })

    fieldnames = ["query_id", "query", "category", "rank", "doc_id",
                  "title", "snippet", "url", "score", "method", "relevant"]
    with open(CANDIDATES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = carried_over + new_unlabeled
    print(f"Wrote {total} candidates to {CANDIDATES_CSV}")
    print(f"  Carried over (already labeled): {carried_over}")
    print(f"  New (need labeling):            {new_unlabeled}")
    if new_unlabeled:
        print(f"Fill in the 'relevant' column (1/0) for the {new_unlabeled} new rows, then re-run without --generate-candidates.")
    else:
        print("All candidates already labeled — run without --generate-candidates to evaluate.")


# main evaluation

def evaluate(queries):
    labeled = [q for q in queries if q.get("relevant_doc_ids")]
    if not labeled:
        print(
            "No relevance judgments found in queries.json.\n"
            "Run with --generate-candidates first, fill in the 'relevant' column,\n"
            "then copy the doc_ids back into queries.json as relevant_doc_ids."
        )
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for cfg in CONFIGS:
        p_scores, r_scores, ap_scores, ndcg_scores = [], [], [], []

        for q in labeled:
            relevant = set(q["relevant_doc_ids"])
            # use full corpus size for MAP so we don't truncate the ranking
            ranked_full = run_config(q["query"], cfg, top_k=CORPUS_SIZE)
            ranked_k = ranked_full[:K]

            p    = precision_at_k(ranked_k, relevant, K)
            r    = recall_at_k(ranked_k, relevant, K)
            ap   = average_precision(ranked_full, relevant)
            ndcg = ndcg_at_k(ranked_k, relevant, K)

            p_scores.append(p)
            r_scores.append(r)
            ap_scores.append(ap)
            ndcg_scores.append(ndcg)

            all_rows.append({
                "config":   cfg["name"],
                "query_id": q["id"],
                "query":    q["query"],
                "category": q["category"],
                "P@10":     round(p, 4),
                "R@10":     round(r, 4),
                "AP":       round(ap, 4),
                "NDCG@10":  round(ndcg, 4),
            })

        print(
            f"{cfg['name']:14s}  "
            f"P@{K}={np.mean(p_scores):.3f}  "
            f"R@{K}={np.mean(r_scores):.3f}  "
            f"MAP={np.mean(ap_scores):.3f}  "
            f"NDCG@{K}={np.mean(ndcg_scores):.3f}"
        )

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(METRICS_CSV, index=False)
    print(f"\nPer-query breakdown saved to {METRICS_CSV}")

    print("\n── By category ──────────────────────────────────────")
    for cat in sorted(df_out["category"].unique()):
        sub = df_out[df_out["category"] == cat]
        print(f"\n  {cat}")
        for cfg in CONFIGS:
            c = sub[sub["config"] == cfg["name"]]
            if c.empty:
                continue
            print(
                f"    {cfg['name']:14s}  "
                f"P@{K}={c['P@10'].mean():.3f}  "
                f"MAP={c['AP'].mean():.3f}  "
                f"NDCG@{K}={c['NDCG@10'].mean():.3f}"
            )


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RFP search engine")
    p.add_argument("--data",    default=str(ROOT / "rfp.csv"), help="Path to rfp.csv")
    p.add_argument("--queries", default=str(QUERIES_FILE),     help="Path to queries.json")
    p.add_argument("--generate-candidates", action="store_true",
                   help="Write top-30 results per query to candidates.csv for labeling")
    return p.parse_args()


def main():
    args = parse_args()

    queries_path = Path(args.queries)
    if not queries_path.exists():
        sys.exit(f"queries file not found: {queries_path}")

    with open(queries_path, encoding="utf-8") as f:
        data = json.load(f)
    queries = data["queries"]

    import main as m
    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"data file not found: {data_path}")

    m.DATA_FILE = data_path
    print(f"Loading indexes from {data_path} ...")
    m.init_indexes()
    print()

    if args.generate_candidates:
        generate_candidates(queries)
    else:
        print(f"Evaluating {len([q for q in queries if q.get('relevant_doc_ids')])} labeled queries "
              f"across {len(CONFIGS)} configs  (K={K})\n")
        print(f"{'config':14s}  {'P@10':>6}  {'R@10':>6}  {'MAP':>6}  {'NDCG@10':>8}")
        print("─" * 55)
        evaluate(queries)


if __name__ == "__main__":
    main()
