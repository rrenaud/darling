import json
import sys
from pathlib import Path

def add_ids(input_path: str, output_path: str) -> None:
    """Read `input_path`, add an `id` key to every JSON object, and
    write results to `output_path`."""
    in_path, out_path = Path(input_path), Path(output_path)

    with in_path.open("r", encoding="utf-8") as infile, \
         out_path.open("w", encoding="utf-8") as outfile:

        for idx, line in enumerate(infile):
            # Skip completely blank lines (often accidental in JSONL).
            if not line.strip():
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"Line {idx}: not valid JSON – {e}")

            # Preserve any existing value by renaming it (optional).
            if "id" in obj:
                obj["_old_id"] = obj.pop("id")

            obj["id"] = idx
            outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✅  Added id 0‑{idx} and wrote {out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(
            "Usage:\n"
            "    python add_id.py input.jsonl output.jsonl"
        )
    add_ids(sys.argv[1], sys.argv[2])

