#!/usr/bin/env python3
"""
Query a local vLLM server with prompts from the DeepScaleR preview dataset.

The script loads `agentica-org/DeepScaleR-Preview-Dataset` from Hugging Face,
sends each question to the provided vLLM REST endpoint, collects multiple
rollouts, and records whether each rollout matches the reference answer.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

import requests
from requests import exceptions as req_exc
from datasets import load_dataset
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query local vLLM server with DeepScaleR prompts."
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("VLLM_HOST", "127.0.0.1"),
        help="Hostname for the local vLLM server (default: %(default)s).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("VLLM_PORT", 8000)),
        help="Port for the local vLLM server (default: %(default)s).",
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("VLLM_MODEL_NAME", "Qwen3-4B"),
        help="Name of the model being evaluated (recorded in the output).",
    )
    parser.add_argument(
        "--dataset",
        default="agentica-org/DeepScaleR-Preview-Dataset",
        help="Dataset name or path to load via datasets.load_dataset.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to evaluate (default: %(default)s).",
    )
    parser.add_argument(
        "--question-field",
        default=None,
        help="Column name for the question. Auto-detected if not provided.",
    )
    parser.add_argument(
        "--answer-field",
        default=None,
        help="Column name for the answer. Auto-detected if not provided.",
    )
    parser.add_argument(
        "--id-field",
        default=None,
        help="Optional column name to use as the question identifier.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature passed to vLLM (default: %(default)s).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling value (default: %(default)s).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate per rollout.",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=8,
        help="Number of rollouts to request per prompt (default: %(default)s).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of dataset samples to process.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("deepscaler_vllm_results.jsonl"),
        help="Path to the JSONL file where results will be appended.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds for the vLLM server.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional dataset revision (tag/branch/commit) to load.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to datasets.load_dataset.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Overwrite the output file instead of appending.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debugging details for failed requests.",
    )
    parser.add_argument(
        "--endpoint",
        choices=("auto", "generate", "completions", "chat_completions"),
        default="auto",
        help="vLLM API endpoint to use. 'auto' tries multiple options.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """Lowercase, strip whitespace, and remove trailing punctuation for comparison."""
    text = text.strip()
    if not text:
        return text
    # Keep simple punctuation trimming while preserving things like minus signs.
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\n\r")
    text = text.lower()
    text = text.strip(" .,:;!?\"'`")
    return text


_FINAL_ANSWER_PATTERNS = [
    re.compile(r"final\s+answer\s*[:\-]\s*(.+)", re.IGNORECASE),
    re.compile(r"answer\s*[:\-]\s*(.+)", re.IGNORECASE),
    re.compile(r"result\s*[:\-]\s*(.+)", re.IGNORECASE),
]


def extract_candidate_answer(response: str) -> str:
    """Attempt to extract the final answer from a model response."""
    for pattern in _FINAL_ANSWER_PATTERNS:
        match = pattern.search(response)
        if match:
            return match.group(1).strip()
    # Fallback: use the last non-empty line.
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    return lines[-1] if lines else response.strip()


def compute_correctness(
    response: str, reference: Optional[str | Iterable[str]]
) -> Optional[bool]:
    """Return True/False when comparison is possible, else None."""
    if reference is None:
        return None

    candidate = extract_candidate_answer(response)
    cand_norm = normalize_text(candidate)
    if isinstance(reference, str):
        ref_norm = normalize_text(reference)
        return cand_norm == ref_norm

    try:
        refs = list(reference)
    except TypeError:
        return None

    normalized_refs = {normalize_text(ref) for ref in refs}
    if not normalized_refs:
        return None
    return cand_norm in normalized_refs


def detect_field(example: dict[str, object], candidates: Iterable[str]) -> Optional[str]:
    for name in candidates:
        if name in example:
            return name
    return None


def prepare_dataset(
    args: argparse.Namespace,
) -> tuple[Iterable[dict[str, object]], Optional[int], str, str, Optional[str]]:
    load_kwargs = {}
    if args.revision:
        load_kwargs["revision"] = args.revision
    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True

    dataset = load_dataset(args.dataset, split=args.split, **load_kwargs)
    total = len(dataset)

    example = dataset[0]
    question_field = args.question_field or detect_field(
        example, ("question", "prompt", "input", "problem")
    )
    answer_field = args.answer_field or detect_field(
        example, ("answer", "answers", "solution", "target", "label")
    )

    if question_field is None:
        raise ValueError(
            "Could not auto-detect the question field. Please provide --question-field."
        )
    if answer_field is None:
        raise ValueError(
            "Could not auto-detect the answer field. Please provide --answer-field."
        )

    id_field = args.id_field
    if id_field and id_field not in example:
        raise ValueError(f"ID field '{id_field}' not present in the dataset.")

    return dataset, total, question_field, answer_field, id_field


def build_payload(endpoint: str, args: argparse.Namespace, prompt: str) -> dict[str, object]:
    if endpoint == "generate":
        return {
            "model": args.model_name,
            "prompt": prompt,
            "n": args.num_rollouts,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        }
    if endpoint == "completions":
        return {
            "model": args.model_name,
            "prompt": prompt,
            "n": args.num_rollouts,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        }
    if endpoint == "chat_completions":
        return {
            "model": args.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "n": args.num_rollouts,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        }
    raise ValueError(f"Unknown endpoint kind '{endpoint}'")


def extract_outputs(response_json: dict[str, object]) -> list[str]:
    if "data" in response_json:
        data = response_json["data"]
        if isinstance(data, list) and data:
            entry = data[0]
            if isinstance(entry, dict) and "outputs" in entry:
                outputs = entry["outputs"]
                if isinstance(outputs, list):
                    return [str(item.get("text", "")) for item in outputs]
    if "outputs" in response_json and isinstance(response_json["outputs"], list):
        return [str(item.get("text", "")) for item in response_json["outputs"]]
    if "choices" in response_json and isinstance(response_json["choices"], list):
        outputs: list[str] = []
        for choice in response_json["choices"]:
            if not isinstance(choice, dict):
                continue
            if "text" in choice:
                outputs.append(str(choice["text"]))
            elif "message" in choice and isinstance(choice["message"], dict):
                outputs.append(str(choice["message"].get("content", "")))
        if outputs:
            return outputs
    if "text" in response_json:
        text = response_json["text"]
        if isinstance(text, list):
            return [str(t) for t in text]
        return [str(text)]
    raise ValueError(f"Unexpected vLLM response format: {response_json}")


def try_query_vllm(
    args: argparse.Namespace, prompt: str
) -> tuple[list[str], str]:
    endpoint_order = {
        "generate": [("/generate", "generate")],
        "completions": [("/v1/completions", "completions")],
        "chat_completions": [("/v1/chat/completions", "chat_completions")],
        "auto": [
            ("/generate", "generate"),
            ("/v1/completions", "completions"),
            ("/v1/chat/completions", "chat_completions"),
        ],
    }
    attempts = endpoint_order[args.endpoint]

    last_error: Optional[Exception] = None
    for path, kind in attempts:
        payload = build_payload(kind, args, prompt)
        try:
            response = requests.post(
                f"http://{args.host}:{args.port}{path}",
                json=payload,
                timeout=args.timeout,
            )
            if response.status_code == 404:
                last_error = req_exc.HTTPError(
                    f"{response.status_code} {response.reason}", response=response
                )
                continue
            response.raise_for_status()
            outputs = extract_outputs(response.json())
            return outputs, path
        except Exception as exc:  # pylint: disable=broad-except
            last_error = exc
            if args.verbose:
                print(f"[DEBUG] Endpoint {path} failed: {exc}", file=sys.stderr)
                if isinstance(exc, req_exc.HTTPError) and exc.response is not None:
                    print(f"[DEBUG] Response body: {exc.response.text}", file=sys.stderr)
    if last_error:
        raise last_error
    raise RuntimeError("All endpoint attempts failed without raising an exception.")


def main() -> int:
    args = parse_args()
    dataset, total, question_field, answer_field, id_field = prepare_dataset(args)

    mode = "w" if args.overwrite_output else "a"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with args.output.open(mode, encoding="utf-8") as writer:
        progress = tqdm(
            dataset,
            total=min(total, args.max_samples) if args.max_samples else total,
            desc="Querying vLLM",
        )
        for entry in progress:
            if args.max_samples and count >= args.max_samples:
                break

            question = entry[question_field]
            answer = entry.get(answer_field)
            if not isinstance(question, str):
                question = str(question)

            last_correct: Optional[bool] = None
            endpoint_used: Optional[str] = None
            try:
                outputs, endpoint_used = try_query_vllm(args, question)
            except Exception as exc:  # pylint: disable=broad-except
                progress.write(f"[WARN] Failed to query prompt index {count}: {exc}")
                continue

            for idx, output_text in enumerate(outputs):
                is_correct = compute_correctness(output_text, answer)
                # Track the most recent correctness value that is not None for display.
                if is_correct is not None:
                    last_correct = is_correct
                record = {
                    "model": args.model_name,
                    "question_index": count,
                    "rollout_index": idx,
                    "question": question,
                    "response": output_text,
                    "reference_answer": answer,
                    "predicted_answer": extract_candidate_answer(output_text),
                    "is_correct": is_correct,
                    "endpoint": endpoint_used,
                }
                if id_field:
                    record["question_id"] = entry[id_field]
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            writer.flush()

            count += 1
            progress.set_postfix(correct=last_correct, endpoint=endpoint_used, refresh=False)

    print(f"Saved {count} prompts to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
