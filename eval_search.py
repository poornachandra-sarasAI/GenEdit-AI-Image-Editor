"""Evaluate GenEdit's semantic image search.

Usage:
    python eval_search.py eval/queries.json --k 1 3 5

queries.json is a list of labelled queries. Each lists the image(s) that should be
returned, by image ID or by the title you gave the image when uploading it:

    [
      {"query": "dog playing on a beach", "relevant": ["beach-dog"]},
      {"query": "red sports car",         "relevant": ["car-1", "car-2"]}
    ]

Reports Recall@k (share of queries with at least one relevant image in the top k)
and MRR (mean reciprocal rank of the first relevant image), plus the misses.
"""
import argparse
import json
import sys

from ai_utils import embed_text
from storage import ensure_setup, list_images, search_by_embedding


def resolve(labels, title_to_ids):
    ids = set()
    for label in labels:
        ids.update(title_to_ids.get(label, [label]))  # unknown title -> treat as an ID
    return ids


def evaluate(queries, ks, embed=embed_text):
    title_to_ids = {}
    for img in list_images():
        title_to_ids.setdefault(img["title"], []).append(img["id"])

    max_k = max(ks)
    hits = {k: 0 for k in ks}
    rr_total = 0.0
    misses = []
    for q in queries:
        relevant = resolve(q["relevant"], title_to_ids)
        ranked = [img_id for img_id, _, _ in search_by_embedding(embed(q["query"]), top_k=max_k)]
        rank = next((i + 1 for i, img_id in enumerate(ranked) if img_id in relevant), None)
        for k in ks:
            hits[k] += rank is not None and rank <= k
        rr_total += 1.0 / rank if rank else 0.0
        if rank != 1:
            misses.append((q["query"], rank))

    n = len(queries)
    return {f"recall@{k}": hits[k] / n for k in ks} | {"mrr": rr_total / n, "n": n}, misses


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("queries", help="Path to labelled queries JSON")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    args = parser.parse_args()

    ensure_setup()
    with open(args.queries) as f:
        queries = json.load(f)
    if not queries:
        sys.exit("No queries found.")

    metrics, misses = evaluate(queries, sorted(args.k))
    print(f"Evaluated {metrics.pop('n')} queries")
    for name, value in metrics.items():
        print(f"  {name:>9}: {value:.3f}")
    if misses:
        print("\nQueries where the top result was not relevant (rank of first relevant hit):")
        for query, rank in misses:
            print(f"  - {query!r}: {rank if rank else f'not in top {max(args.k)}'}")


if __name__ == "__main__":
    main()
