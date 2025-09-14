import argparse, json, sys
from collections import defaultdict

def load_jsonl(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed JSON at {path}:{ln}: {e}", file=sys.stderr)
                continue
            _id = obj.get("id")
            if _id is None:
                print(f"[WARN] Missing 'id' at {path}:{ln}; skipping", file=sys.stderr)
                continue
            # last one wins if duplicates
            data[_id] = {
                "distinct": obj.get("distinct"),
                "prompt": obj.get("prompt"),
                "model": obj.get("model"),
                "raw": obj,
            }
    return data

def main():
    ap = argparse.ArgumentParser(
        description="Find IDs where partition_multiplicative_no_norm has much more 'distinct' than quality_only."
    )
    ap.add_argument("--quality", required=True,
                    help="Path to quality_only JSONL (e.g., tianjian_wildchat10k_quality_only_.../partitions.jsonl)")
    ap.add_argument("--multiplicative", required=True,
                    help="Path to partition_multiplicative_no_norm JSONL (e.g., llama_70b_wildchat_wildchat10k_quality_partition_multiplicative_no_norm_.../partitions.jsonl)")
    ap.add_argument("--ratio", type=float, default=1.5,
                    help="Minimum ratio (multiplicative / quality) to count as 'much more' (default: 1.5)")
    ap.add_argument("--diff", type=int, default=2,
                    help="Minimum absolute difference (multiplicative - quality) (default: 2)")
    ap.add_argument("--verbose", action="store_true",
                    help="Print extra info (distinct counts, ratio, diff) instead of just IDs")
    args = ap.parse_args()

    q = load_jsonl(args.quality)
    m = load_jsonl(args.multiplicative)

    common_ids = set(q.keys()) & set(m.keys())
    if not common_ids:
        print("[WARN] No overlapping IDs between the two files.", file=sys.stderr)

    results = []
    for _id in common_ids:
        qd = q[_id].get("distinct")
        md = m[_id].get("distinct")

        # Skip if either missing or not numeric
        if not isinstance(qd, (int, float)) or not isinstance(md, (int, float)):
            continue

        diff = md - qd
        if qd == 0:
            ratio = float("inf") if md > 0 else 1.0
        else:
            ratio = md / qd

        if diff >= args.diff:
            results.append((_id, qd, md, ratio, diff))

    # Sort by ratio (desc), then diff (desc)
    results.sort(key=lambda x: (x[3], x[4]), reverse=True)

    if args.verbose:
        for _id, qd, md, ratio, diff in results:
            print(f"{_id}\tquality={qd}\tmultiplicative={md}\tratio={ratio:.2f}\tdiff={diff}")
    else:
        for _id, *_ in results:
            print(_id)

if __name__ == "__main__":
    main()

