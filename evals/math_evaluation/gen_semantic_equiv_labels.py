#!/usr/bin/env python3
"""
Utility to build a semantic equivalence dataset for math reasoning responses by
querying the GPToss model served through DeepInfra.

The script can read trace pairs from either a local JSONL file or directly from
the OpenR1-Math-220k dataset released on Hugging Face (default behaviour).
The dataset exposes fields such as ``problem``, ``solution``, and ``answer``
alongside a ``generations`` list containing multiple model responses for the
same prompt. The script pairs distinct generations from each prompt, or falls
back to pre-paired fields when available.

When using JSONL, each record should look like:
{
    "id": "unique identifier",
    "prompt": "original math problem prompt",
    "trace_a": "first reasoning response",
    "trace_b": "second reasoning response"
}

The output JSONL will add the model response and a parsed verdict per record.
"""
from __future__ import annotations

import argparse
import json
import logging
import concurrent.futures
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from itertools import combinations
from typing import Any, Dict, Iterable, Iterator, Optional

try:
    from openai import OpenAI
    from openai import OpenAIError
except ImportError:  # pragma: no cover - handled at runtime
    OpenAI = None  # type: ignore
    OpenAIError = Exception  # type: ignore


DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"  # https://deepinfra.com/docs/openai_api
DEFAULT_API_KEY_PATH = Path("/home/rrenaud/deepinfra_api_key.txt")
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3.2-Exp"
DEFAULT_HF_DATASET = "open-r1/OpenR1-Math-220k"
DEFAULT_PROMPT_FIELD = "problem"
DEFAULT_GENERATIONS_FIELD = "generations"
# JSON schema describing the structured response we expect.
SEMANTIC_EQUIVALENCE_SCHEMA = {
    "name": "SemanticEquivalence",
    "schema": {
        "type": "object",
        "properties": {
            "equivalent": {"type": "boolean"},
            "confidence": {"type": "number"},
            "justification": {"type": "string"},
        },
        "required": ["equivalent", "confidence", "justification"],
    },
}

SYSTEM_PROMPT = "You are an expert annotator who must decide if two math reasoning responses are semantically equivalent. Output JSON only."

USER_PROMPT_TEMPLATE = """You are given the original prompt and two model-generated responses. Determine whether these responses are semantically equivalent, based on whether reading the second response would provide the reader with substantially new or different information compared to the first.
Original prompt: \"\"\" {prompt} \"\"\"
Generation 0: \"\"\" {gen0} \"\"\"
Generation 1: \"\"\" {gen1} \"\"\"
Question: Are Generation 0 and Generation 1 semantically equivalent?
Think briefly step-by-step:
Core Meaning: Do both responses essentially communicate the same key points or concepts?
Additional Information: Would reading the second response significantly add new ideas, examples, or important details beyond the first?
Briefly provide your reasoning, then explicitly conclude:
[[Yes]]: The second response does not significantly add new information or insights.
[[No]]: The second response introduces meaningful new or distinct ideas, insights, or details.

Return a JSON object with:
- "equivalent": true if [[Yes]], false if [[No]]
- "confidence": float in [0, 1]
- "justification": short explanation of your decision
"""


@dataclass
class TracePair:
    """Container for a pair of responses associated with the same math prompt."""

    id: str
    prompt: str
    trace_a: str
    trace_b: str
    metadata: Dict[str, Any]

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "TracePair":
        id_ = payload.get("id") or payload.get("example_id") or ""
        if not id_:
            raise ValueError("Trace pair is missing an 'id' field.")
        prompt = payload.get("prompt") or payload.get("question") or payload.get("problem") or ""
        if not prompt:
            raise ValueError(f"Trace pair {id_!r} is missing a 'prompt' field.")
        trace_a = payload.get("trace_a") or payload.get("trajectory_a") or payload.get("trace1")
        trace_b = payload.get("trace_b") or payload.get("trajectory_b") or payload.get("trace2")
        if trace_a is None or trace_b is None:
            raise ValueError(f"Trace pair {id_!r} is missing trace content.")
        metadata_keys = {
            "id",
            "example_id",
            "prompt",
            "question",
            "problem",
            "trace_a",
            "trace_b",
            "trajectory_a",
            "trajectory_b",
            "trace1",
            "trace2",
        }
        metadata = {k: v for k, v in payload.items() if k not in metadata_keys}
        return cls(id=id_, prompt=str(prompt), trace_a=str(trace_a), trace_b=str(trace_b), metadata=metadata)


@dataclass
class ClassificationResult:
    """Structured prediction returned by the model."""

    equivalent: Optional[bool]
    confidence: Optional[float]
    justification: str
    raw_response: str
    usage: Optional[Dict[str, Any]]

    def to_jsonl_record(self, pair: TracePair, model_name: str) -> Dict[str, Any]:
        record = {
            "id": pair.id,
            "model": model_name,
            "equivalent": self.equivalent,
            "confidence": self.confidence,
            "justification": self.justification,
            "raw_response": self.raw_response,
            "usage": self.usage,
        }
        if pair.metadata:
            record["metadata"] = pair.metadata
        return record


def load_api_key(path: Path) -> str:
    try:
        api_key = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"DeepInfra API key file not found at {path}") from exc
    if not api_key:
        raise ValueError(f"DeepInfra API key file at {path} is empty.")
    return api_key


def build_openai_client(api_key: str) -> "OpenAI":
    if OpenAI is None:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "The 'openai' package is required to call DeepInfra's OpenAI-compatible API. "
            "Install it with `pip install openai`."
        )
    return OpenAI(api_key=api_key, base_url=DEEPINFRA_BASE_URL)


def read_pairs(path: Path, limit: Optional[int] = None) -> Iterator[TracePair]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            yield TracePair.from_json(payload)
            if limit is not None and idx >= limit:
                break


def iter_pairs_from_hf(
    dataset_id: str,
    split: str,
    config: Optional[str],
    revision: Optional[str],
    id_field: str,
    prompt_field: str,
    gen0_field: Optional[str],
    gen1_field: Optional[str],
    generations_field: Optional[str],
    max_pairs_per_prompt: Optional[int],
    limit: Optional[int],
) -> Iterator[TracePair]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required to pull datasets from Hugging Face. "
            "Install it with `pip install datasets`."
        ) from exc

    dataset = load_dataset(
        dataset_id,
        name=config,
        split=split,
        revision=revision,
    )

    skip_fields = {id_field, prompt_field}
    if gen0_field:
        skip_fields.add(gen0_field)
    if gen1_field:
        skip_fields.add(gen1_field)
    if generations_field:
        skip_fields.add(generations_field)

    produced = 0

    for row_idx, row in enumerate(dataset):
        record_id = row.get(id_field)
        if record_id is None:
            record_id = str(row_idx)
        prompt = str(row.get(prompt_field, ""))
        metadata = {k: v for k, v in row.items() if k not in skip_fields}

        if generations_field:
            raw_generations = row.get(generations_field) or []
            if not isinstance(raw_generations, list):
                continue
            indexed_generations = [
                (gen_idx, str(gen))
                for gen_idx, gen in enumerate(raw_generations)
                if gen is not None and str(gen).strip()
            ]
            if len(indexed_generations) < 2:
                continue
            pair_counter = 0
            for (i_idx, trace_a), (j_idx, trace_b) in combinations(indexed_generations, 2):
                payload: Dict[str, Any] = {
                    "id": f"{record_id}-{i_idx}-{j_idx}",
                    "prompt": prompt,
                    "trace_a": str(trace_a),
                    "trace_b": str(trace_b),
                    "generation_indices": [i_idx, j_idx],
                }
                payload.update(metadata)
                yield TracePair.from_json(payload)
                produced += 1
                pair_counter += 1
                if limit is not None and produced >= limit:
                    return
                if max_pairs_per_prompt is not None and pair_counter >= max_pairs_per_prompt:
                    break
        else:
            if not gen0_field or not gen1_field:
                raise ValueError(
                    "Either a generations field or both gen0/gen1 fields must be provided for Hugging Face datasets."
                )
            trace_a = row.get(gen0_field, "")
            trace_b = row.get(gen1_field, "")
            if trace_a is None or trace_b is None:
                continue
            payload = {
                "id": str(record_id),
                "prompt": prompt,
                "trace_a": str(trace_a),
                "trace_b": str(trace_b),
            }
            payload.update(metadata)
            yield TracePair.from_json(payload)
            produced += 1
            if limit is not None and produced >= limit:
                return


def build_request_params(pair: TracePair, temperature: float) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    prompt=pair.prompt,
                    gen0=pair.trace_a,
                    gen1=pair.trace_b,
                ),
            },
        ],
        "temperature": temperature,
        "response_format": {
            "type": "json_schema",
        "json_schema": SEMANTIC_EQUIVALENCE_SCHEMA,
        },
    }


def extract_choice_content(response: Dict[str, Any]) -> str:
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unexpected DeepInfra response schema: {response}") from exc
    parsed = message.get("parsed")
    if parsed is not None:
        return json.dumps(parsed)
    content = message.get("content")
    if isinstance(content, list):
        # OpenAI client may return content as a list of dicts with 'type'/'text'.
        parts = []
        for item in content:
            if isinstance(item, dict):
                part = item.get("text") or item.get("content") or ""
                parts.append(str(part))
            else:
                parts.append(str(item))
        content = "\n".join(parts)
    if content is None:
        raise ValueError(f"No content found in response message: {response}")
    return str(content)


def parse_model_json(payload: str) -> Dict[str, Any]:
    text = payload.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Attempt to salvage JSON substring.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            logging.warning("Failed to parse model JSON payload: %s", text)
            return {}
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            logging.warning("Failed to parse JSON substring from model payload: %s", text)
            return {}


def classify_pair(
    client: "OpenAI",
    model: str,
    pair: TracePair,
    temperature: float,
    max_retries: int,
    retry_delay: float,
    max_tokens: Optional[int],
) -> ClassificationResult:
    request_params = build_request_params(pair, temperature)

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=request_params["messages"],
                temperature=request_params["temperature"],
                response_format=request_params["response_format"],
                max_tokens=max_tokens,
            )
            response_dict = response.model_dump()
            content = extract_choice_content(response_dict)
            parsed = parse_model_json(content)
            usage = response_dict.get("usage")
            return ClassificationResult(
                equivalent=parsed.get("equivalent"),
                confidence=parsed.get("confidence"),
                justification=parsed.get("justification", ""),
                raw_response=content,
                usage=usage,
            )
        except (OpenAIError, ValueError) as exc:
            last_error = exc
            logging.warning(
                "Attempt %d/%d failed for pair %s: %s",
                attempt,
                max_retries,
                pair.id,
                exc,
            )
            if attempt < max_retries:
                time.sleep(retry_delay)
    raise RuntimeError(f"Failed to classify pair {pair.id} after {max_retries} attempts") from last_error


def ensure_output_path(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output file {path} already exists. Use --overwrite to replace it.")
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate semantic equivalence labels for pairs of math reasoning responses using GPToss via DeepInfra."
    )
    parser.add_argument(
        "--pairs-file",
        type=Path,
        help="Path to a JSONL file containing math reasoning prompt/response pairs.",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default=DEFAULT_HF_DATASET,
        help=(
            "Hugging Face dataset repo ID to pull trace pairs from "
            f"(default: {DEFAULT_HF_DATASET}; pass empty string to disable)."
        ),
    )
    parser.add_argument(
        "--hf-config",
        type=str,
        default=None,
        help="Optional dataset configuration name when pulling from Hugging Face.",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default="train",
        help="Dataset split to load from Hugging Face (default: train).",
    )
    parser.add_argument(
        "--hf-revision",
        type=str,
        default=None,
        help="Dataset revision or commit SHA to use from Hugging Face.",
    )
    parser.add_argument(
        "--hf-id-field",
        type=str,
        default="uuid",
        help="Field name containing the example identifier in the Hugging Face dataset (default: uuid).",
    )
    parser.add_argument(
        "--hf-prompt-field",
        type=str,
        default=DEFAULT_PROMPT_FIELD,
        help=f"Field name containing the original prompt in the Hugging Face dataset (default: {DEFAULT_PROMPT_FIELD}).",
    )
    parser.add_argument(
        "--hf-gen0-field",
        type=str,
        default="",
        help="Optional field name for the first response when the dataset stores paired generations.",
    )
    parser.add_argument(
        "--hf-gen1-field",
        type=str,
        default="",
        help="Optional field name for the second response when the dataset stores paired generations.",
    )
    parser.add_argument(
        "--hf-generations-field",
        type=str,
        default=DEFAULT_GENERATIONS_FIELD,
        help=(
            "Field containing multiple generations per prompt in the Hugging Face dataset "
            f"(default: {DEFAULT_GENERATIONS_FIELD}; set to empty string to disable)."
        ),
    )
    parser.add_argument(
        "--hf-max-pairs-per-prompt",
        type=int,
        default=None,
        help="If provided, restrict the number of sampled pairs per prompt when using --hf-generations-field.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Where to write the JSONL with classification results.",
    )
    parser.add_argument(
        "--api-key-path",
        type=Path,
        default=DEFAULT_API_KEY_PATH,
        help=f"Path to the DeepInfra API key (default: {DEFAULT_API_KEY_PATH}).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"DeepInfra model identifier to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature passed to the model (default: 0.0).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate in each response (default: model-defined).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If provided, only process the first N pairs.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Number of times to retry a failed API call (default: 3).",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=5.0,
        help="Seconds to wait between retries (default: 5).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional delay between successful API calls to respect rate limits.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting the output file if it exists.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=16,
        help="Number of concurrent API requests to issue (default: 16).",
    )
    args = parser.parse_args(argv)

    configure_logging(args.verbose)

    hf_dataset = (args.hf_dataset or "").strip() or None
    hf_generations_field = (args.hf_generations_field or "").strip() or None
    hf_gen0_field = (args.hf_gen0_field or "").strip() or None
    hf_gen1_field = (args.hf_gen1_field or "").strip() or None

    if hf_generations_field is None and (hf_gen0_field is None or hf_gen1_field is None) and not args.pairs_file:
        parser.error(
            "When using Hugging Face datasets you must supply --hf-generations-field or both --hf-gen0-field/--hf-gen1-field."
        )

    if not args.pairs_file and not hf_dataset:
        parser.error("Either --pairs-file or --hf-dataset must be provided.")

    try:
        api_key = load_api_key(args.api_key_path)
    except (FileNotFoundError, ValueError) as exc:
        logging.error("%s", exc)
        return 1

    try:
        client = build_openai_client(api_key)
    except ImportError as exc:
        logging.error("%s", exc)
        return 1

    try:
        ensure_output_path(args.output_file, args.overwrite)
    except FileExistsError as exc:
        logging.error("%s", exc)
        return 1

    if args.pairs_file:
        pairs_iter = read_pairs(args.pairs_file, args.limit)
    elif hf_dataset:
        logging.info(
            "Loading pairs from Hugging Face dataset %s (split=%s, config=%s, revision=%s)",
            hf_dataset,
            args.hf_split,
            args.hf_config,
            args.hf_revision,
        )
        try:
            pairs_iter = iter_pairs_from_hf(
                dataset_id=hf_dataset,
                split=args.hf_split,
                config=args.hf_config,
                revision=args.hf_revision,
                id_field=args.hf_id_field,
                prompt_field=args.hf_prompt_field,
                gen0_field=hf_gen0_field,
                gen1_field=hf_gen1_field,
                generations_field=hf_generations_field,
                max_pairs_per_prompt=args.hf_max_pairs_per_prompt,
                limit=args.limit,
            )
        except Exception as exc:
            logging.error("Failed to load dataset from Hugging Face: %s", exc)
            return 1
    else:
        parser.error("Either --pairs-file or --hf-dataset must be provided.")

    processed = 0
    futures: Dict[concurrent.futures.Future, TracePair] = {}

    with args.output_file.open("w", encoding="utf-8") as sink, concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, args.parallelism)
    ) as executor:
        def submit_pair(pair: TracePair) -> None:
            future = executor.submit(
                classify_pair,
                client,
                args.model,
                pair,
                args.temperature,
                args.max_retries,
                args.retry_delay,
                args.max_tokens,
            )
            futures[future] = pair

        def drain_completed(block: bool = False) -> None:
            nonlocal processed
            if block:
                iterator = concurrent.futures.as_completed(list(futures.keys()))
            else:
                iterator = [future for future in list(futures.keys()) if future.done()]

            for future in iterator:
                pair = futures.pop(future, None)
                if pair is None:
                    continue
                try:
                    result = future.result()
                except Exception as exc:
                    logging.error("Classification failed for pair %s: %s", pair.id, exc)
                    continue

                record = result.to_jsonl_record(pair, args.model)
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                processed += 1
                logging.info("Classified pair %s (total %d)", pair.id, processed)
                if args.sleep:
                    time.sleep(args.sleep)

        for pair in pairs_iter:
            submit_pair(pair)
            if len(futures) >= args.parallelism:
                drain_completed(block=False)

        while futures:
            drain_completed(block=True)

    logging.info("Finished processing %d pairs.", processed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
