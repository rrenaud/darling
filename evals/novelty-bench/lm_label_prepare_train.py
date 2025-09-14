from __future__ import annotations

import argparse
import json
import random
import sys
import time
import re
from pathlib import Path
from typing import Dict, List

from openai import AzureOpenAI
import jsonlines

# --------------------------------------------------------------------------- #
# Prompt template for the judge LLM
# --------------------------------------------------------------------------- #
_JUDGE_PROMPT = """
You are given the original prompt and two model-generated responses. Determine whether the two responses use different strategies to solve the problem.

Use the following guidelines:

- Different solution methods: Clearly different approaches (e.g., algebraic vs. geometric, analytical vs. numerical).
- Critical reasoning divergence: Significant differences in key reasoning steps or assumptions, even if final answers match.
- Conceptual differences: Distinct underlying concepts or representations (e.g., probability vs. combinatorics).

**Also label as different if:**  
The two responses share the same general approach but differ meaningfully in specific intermediate steps or manipulations crucial to the solution.

Original prompt:
\"\"\"{prompt}\"\"\"

Generation 0:
\"\"\"{gen0}\"\"\"

Generation 1:
\"\"\"{gen1}\"\"\"

Question: Do Generation 0 and Generation 1 use different strategies? You may first generate a short reasoning, then respond with "[[yes]]" if the generations use different strategies or "[[no]]" if they do not. 
"""

_CONGNITIVE_JUDGE_PROMPT = """You are given a math problem (Original Prompt) and two model-generated solutions (Generation 0 and Generation 1). Classify each response according to the presence of the following cognitive behaviors:

Verification: Explicitly checks intermediate or final results against the conditions or criteria of the problem (e.g., "Checking if x = 2 satisfies the equation...").

Subgoal Setting: Clearly identifies intermediate steps or goals explicitly set to guide toward the final solution (e.g., "First, I'll simplify the equation to isolate x...").

Backtracking: Explicitly reverses or discards previous steps or attempts to explore a new solution path after realizing a previous approach was incorrect or insufficient (e.g., "This approach leads nowhere, let's try another...").

Backward Chaining: Starts reasoning from the desired goal or final result and logically works backward towards initial conditions (e.g., "To obtain the final value of x, let's see what conditions must hold first...").

Provide your classification separately for each generation using the tags:

Verification: <verification>yes/no</verification>

Subgoal Setting: <subgoal>yes/no</subgoal>

Backtracking: <backtracking>yes/no</backtracking>

Backward Chaining: <backward_chaining>yes/no</backward_chaining>

Original Prompt:
\"\"\"{prompt}\"\"\"

Generation 0:
\"\"]"{gen0}\"\"\"

Generation 1:
\"\"\"{gen1}\"\"\"

After classifying each generation, briefly evaluate if Generation 0 and Generation 1 exhibit exactly the same cognitive behaviors. Respond with "[[Yes]]" if they exhibit exactly the same behaviors, or "[[No]]" if they do not.
"""


# --------------------------------------------------------------------------- #
# Azure client helper
# --------------------------------------------------------------------------- #

def make_azure_client(model_name: str) -> AzureOpenAI:
    """Instantiate AzureOpenAI client for the chosen model."""
    _endpoints = {
        "gpt-4o": "https://azure-services-fair-openai1-northcentralus.azure-api.net",
        "o1-mini": "https://azure-services-fair-openai1-westus3.azure-api.net",
        "o3": "https://azure-services-fair-openai1-eastus2n2.azure-api.net",
        "gpt-4-turbo": "https://azure-services-fair-openai1-westus.azure-api.net",
    }
    if model_name not in _endpoints:
        raise ValueError(f"Unknown model '{model_name}'. Choices: {list(_endpoints)}")

    api_key_env = model_name.replace("-", "_")  # e.g. gpt-4-turbo → gpt_4_turbo
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable '{api_key_env}' with API key not set.")

    return AzureOpenAI(
        api_version="2025-02-01-preview",
        api_key=api_key,
        azure_endpoint=_endpoints[model_name],
    )

def make_openai_client(port: int = 8000):
    """Create an OpenAI-compatible client that connects to a local server."""
    # make openai client
    from openai import OpenAI
    return OpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key='token-abc123',  # No key needed for local server
    )

def get_client(model_name: str):
    """Get the appropriate client based on model name."""
    if model_name == "llama":
        return make_openai_client()
    else:
        return make_azure_client(model_name)

# --------------------------------------------------------------------------- #
# Core evaluation logic
# --------------------------------------------------------------------------- #

def judge_pair(client: AzureOpenAI, prompt: str, gen0: str, gen1: str, model_name: str, max_retries: int = 3) -> int:
    """Return 1 if strategies differ, 0 otherwise."""
    user_message = _CONGNITIVE_JUDGE_PROMPT.format(
        prompt=prompt.strip(),
        gen0=gen0.strip(),
        gen1=gen1.strip()
    )
    
    # user_message = _JUDGE_PROMPT.format(
    #     prompt=prompt.strip(),
    #     gen0=gen0.strip(),
    #     gen1=gen1.strip()
    # )
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": user_message}],
                #max_tokens=8,
            )
            content = resp.choices[0].message.content.strip().lower()
            print(content)
            # extract the last [[*]] in the content
            content = content.split("[[")[-1].split("]]")[0].strip()
            print(f"{content}")
            if content == "yes": # different strategies
                return 0
            elif content == "no": # not different strategies
                return 1
        except Exception as e:
            print(f"[Attempt {attempt}/{max_retries}] Error: {e}", file=sys.stderr)
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                print("Max retries reached; defaulting to not similar (0).", file=sys.stderr)
                return 0
    return 0  # fallback


# ...existing code...

def extract_boxed_answer(text: str) -> str:
    """Extract the content from the last \boxed{} in the text."""
    # Find all \boxed{...} patterns
    boxed_pattern = r'\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}'
    matches = re.findall(boxed_pattern, text)
    if matches:
        return matches[-1].strip()  # Return the last match
    return ""
# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Judge strategic similarity for random response pairs.")
    parser.add_argument("input_jsonl", type=Path, help="combined_results.jsonl path")
    parser.add_argument("--judge-model", "-m", default="gpt-4o", help="Azure model name used as judge")
    parser.add_argument("--train-out", default="ds_train.jsonl", help="Output train JSONL filename (written next to input)")
    parser.add_argument("--val-out", default="ds_val.jsonl", help="Output val JSONL filename (written next to input)")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of records for validation set")
    args = parser.parse_args()

    # Initialize judge client
    client = get_client(args.judge_model)

    # Load data
    rows: List[Dict] = []
    with jsonlines.open(args.input_jsonl, "r") as reader:
        for obj in reader:
            rows.append(obj)
    if not rows:
        print("No data found in input.", file=sys.stderr)
        sys.exit(1)


    random.seed(42)

    augmented: List[Dict] = []
    n_similar = 0
    n_diff = 0

    for idx, row in enumerate(rows):
        prompt = row.get("prompt")
        # Collect answer fields (answer1..answer7)
        answers_with_boxed = []
        for i in range(1, 8):
            answer_key = f"answer{i}"
            if answer_key in row:
                answer_text = row[answer_key]
                boxed_answer = extract_boxed_answer(answer_text)
                if boxed_answer:
                    answers_with_boxed.append((answer_text, boxed_answer))
                    
        if len(answers_with_boxed) < 2:
            print(f"Skipping prompt {idx} with insufficient boxed answers: {answers_with_boxed}", file=sys.stderr)
            continue
        
        boxed_groups = {}
        for answer_text, boxed_answer in answers_with_boxed:
            if boxed_answer not in boxed_groups:
                boxed_groups[boxed_answer] = []
            boxed_groups[boxed_answer].append(answer_text)
        
        equivalent_pairs = []
        for boxed_answer, answer_list in boxed_groups.items():
            if len(answer_list) >= 2:
                equivalent_pairs.extend([(answer_list[i], answer_list[j], boxed_answer) 
                                       for i in range(len(answer_list)) 
                                       for j in range(i+1, len(answer_list))])
        
        if not equivalent_pairs:
            print(f"No equivalent pairs found for prompt {idx} with boxed answers: {boxed_groups}", file=sys.stderr)
            continue
        gen0, gen1, shared_boxed_answer = random.choice(equivalent_pairs)
        label = judge_pair(client, prompt, gen0, gen1, args.judge_model)
        if label == 0:
            n_similar += 1
        else:
            n_diff += 1

        augmented.append({
            "id": idx,
            "labeler": args.judge_model,
            "prompt": prompt,
            "generation_0": gen0,
            "generation_1": gen1,
            "similar": label,
        })
        print(f"[{idx}] similar={label}, boxed_answer={shared_boxed_answer}")

    # Shuffle and split
    random.shuffle(augmented)
    val_size = int(len(augmented) * args.val_ratio)
    val_data = augmented[:val_size]
    train_data = augmented[val_size:]

    in_dir = args.input_jsonl.parent
    train_path = in_dir / args.train_out
    val_path = in_dir / args.val_out

    # add model name to output paths before the .jsonl
    train_path = train_path.with_name(train_path.stem + f"_{args.judge_model}.jsonl")
    val_path = val_path.with_name(val_path.stem + f"_{args.judge_model}.jsonl")

    with jsonlines.open(train_path, "w") as writer:
        for obj in train_data:
            writer.write(obj)
    with jsonlines.open(val_path, "w") as writer:
        for obj in val_data:
            writer.write(obj)

    print("--- Summary ---")
    print(f"Total prompts evaluated: {len(augmented)}")
    print(f"Not similar (0): {n_similar}")
    print(f"Similar (1): {n_diff}")
    print(f"Train set → {train_path}  (n={len(train_data)})")
    print(f"Val set   → {val_path}    (n={len(val_data)})")


if __name__ == "__main__":
    import os  # Needed for env lookups inside make_azure_client
    main()
