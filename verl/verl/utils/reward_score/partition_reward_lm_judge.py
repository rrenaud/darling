import asyncio
import functools
import math
from typing import Dict, List, Optional, Tuple

import httpx

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  – adjust HOSTNAME if needed
# ──────────────────────────────────────────────────────────────────────────────
HOSTNAME = "cr1-h200-p5en48xlarge-4"
NUM_SERVERS = 2  # two LM‑judge instances
SERVER_CFGS = [
    {"url": f"http://{HOSTNAME}:{9000 + i}", "model": f"judge_gpu_{i}"}
    for i in range(NUM_SERVERS)
]

MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds, exponential back‑off base

# ──────────────────────────────────────────────────────────────────────────────
# LIGHTWEIGHT CACHES
# ──────────────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=None)
def get_client(base_url: str) -> httpx.AsyncClient:
    """Return a cached AsyncClient for the given base URL."""
    return httpx.AsyncClient(base_url=base_url, timeout=600)

# ──────────────────────────────────────────────────────────────────────────────
# VERY CHEAP LOCAL CHECK FOR TINY ANSWERS
# ──────────────────────────────────────────────────────────────────────────────

def maybe_test_equality(r0: str, r1: str) -> Optional[bool]:
    """If both responses ≤ 5 tokens, decide by exact unigram overlap."""
    u0, u1 = r0.strip().lower().split(), r1.strip().lower().split()
    max_len = max(len(u0), len(u1))
    if max_len <= 5:
        return len(set(u0) & set(u1)) * 2 >= max_len
    return None

# ──────────────────────────────────────────────────────────────────────────────
# REMOTE JUDGE CALL
# ──────────────────────────────────────────────────────────────────────────────

MATH_JUDGE_PROMPT = """
You are given the original prompt and two model-generated responses. Determine whether the two responses use different strategies to solve the problem.

Use the following guidelines:

- Different solution methods: Clearly different approaches (e.g., algebraic vs. geometric, analytical vs. numerical).
- Critical reasoning divergence: Significant differences in key reasoning steps or assumptions, even if final answers match.
- Conceptual differences: Distinct underlying concepts or representations (e.g., probability vs. combinatorics).

**Also label as different if:**  
The two responses share the same general approach but differ meaningfully in specific intermediate steps or manipulations crucial to the solution.

Generation 0:
\"\"\"{gen0}\"\"\"

Generation 1:
\"\"\"{gen1}\"\"\"

Question: Do Generation 0 and Generation 1 use different strategies? You may first generate a short reasoning, then respond with "[[yes]]" if the generations use different strategies or "[[no]]" if they do not. 
"""

async def remote_judge_equivalent(r0: str, r1: str, cfg: Dict) -> bool:
    """Return True if the LM judge says the responses are equivalent."""

    prompt = MATH_JUDGE_PROMPT.format(prompt="", gen0=r0.strip(), gen1=r1.strip())

    payload = {
        "model": cfg["model"],
        "temperature": 0.0,
        "messages": [
            {"role": "user", "content": prompt},
        ],
    }

    for attempt in range(MAX_RETRIES):
        try:
            client = get_client(cfg["url"])
            resp = await client.post("/v1/chat/completions", json=payload)
            resp.raise_for_status()
            content = (
                resp.json()["choices"][0]["message"]["content"].strip().split()
            )
            # extract the last occurence of [[Yes]] or [[No]]
            verdict = content[-1].strip("[]").lower() == "yes" # they are similar

            return verdict

        except (httpx.HTTPError, httpx.TimeoutException, httpx.ConnectError) as e:
            # Refresh client on connection‑level errors
            if isinstance(e, (httpx.ConnectError, httpx.ReadError)):
                get_client.cache_clear()
            if attempt == MAX_RETRIES - 1:
                raise
            await asyncio.sleep(RETRY_DELAY * (2 ** attempt))

    # Should never reach here
    return False

# ──────────────────────────────────────────────────────────────────────────────
# EQUIVALENCE PREDICATE (local heuristic → remote judge)
# ──────────────────────────────────────────────────────────────────────────────
async def equivalence_check(
    prompt: str,  # kept for signature compatibility (unused)
    r0: str,
    r1: str,
    cfg: Dict,
) -> bool:
    eq = maybe_test_equality(r0, r1)
    if eq is not None:
        return eq
    return await remote_judge_equivalent(r0, r1, cfg)

# ──────────────────────────────────────────────────────────────────────────────
# UNION‑FIND PARTITIONING FOR ONE UID, RUN ON A GIVEN SERVER
# ──────────────────────────────────────────────────────────────────────────────
async def process_uid_partition(
    uid: str,
    responses: List[str],
    indices: List[int],
    prompt: str,
    cfg: Dict,
) -> Tuple[str, List[List[int]]]:
    n = len(responses)
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    async def check(i, j):
        same = await equivalence_check(prompt, responses[i], responses[j], cfg)
        return i, j, same

    tasks = [check(i, j) for i in range(n) for j in range(i + 1, n)]
    for i, j, same in await asyncio.gather(*tasks):
        if same:
            union(i, j)

    groups: Dict[int, List[int]] = {}
    for k in range(n):
        root = find(k)
        groups.setdefault(root, []).append(indices[k])

    return uid, list(groups.values())

# ──────────────────────────────────────────────────────────────────────────────
# ASYNC FAN‑OUT ACROSS UID BLOCKS & SERVERS
# ──────────────────────────────────────────────────────────────────────────────
async def partition_async(
    solution_str: List[str],
    uid: List[str],
    prompts: List[str],
) -> Dict[str, List[List[int]]]:
    # Bucket answers by UID
    by_uid_resp, by_uid_idx, by_uid_prompt = {}, {}, {}
    for idx, (ans, u, prm) in enumerate(zip(solution_str, uid, prompts)):
        by_uid_resp.setdefault(u, []).append(ans)
        by_uid_idx.setdefault(u, []).append(idx)
        by_uid_prompt.setdefault(u, prm)

    tasks = []
    for n, (u, resps) in enumerate(by_uid_resp.items()):
        cfg = SERVER_CFGS[n % NUM_SERVERS]
        tasks.append(
            asyncio.create_task(
                process_uid_partition(u, resps, by_uid_idx[u], by_uid_prompt[u], cfg)
            )
        )

    results = await asyncio.gather(*tasks)
    return {u: parts for u, parts in results}

# ──────────────────────────────────────────────────────────────────────────────
# SYNC WRAPPER – REWARD = distinct_count / (TOTAL_RESPONSES‑1)
# ──────────────────────────────────────────────────────────────────────────────

def partition(**kwargs) -> List[float]:
    solution_str: List[str] = kwargs.get("solution_str", [])
    uid: List[str] = kwargs.get("uid")
    prompts: List[str] = kwargs.get("prompts", [])

    if uid is None:
        raise ValueError("uid is required for partition reward function")

    uid_parts = asyncio.run(partition_async(solution_str, uid, prompts))

    uid_to_indices: Dict[str, List[int]] = {}
    for i, u in enumerate(uid):
        uid_to_indices.setdefault(u, []).append(i)

    rewards = [0.0] * len(solution_str)
    for u, parts in uid_parts.items():
        total_for_uid = len(uid_to_indices[u])
        denom = max(total_for_uid - 1, 1)  # avoid ÷0 when only 1 answer
        for grp in parts:
            distinct = total_for_uid - len(grp)
            for idx in grp:
                rewards[idx] = distinct / denom

    return rewards


def partition_sigmoid(**kwargs) -> List[float]:
    """Sigmoid‑scaled version of `partition`."""
    return [1 / (1 + math.exp(-r)) for r in partition(**kwargs)]

# ──────────────────────────────────────────────────────────────────────────────
# SELF‑TEST (uses mocked LM‑judge that mirrors exact string match for demo)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    async def mock_judge_equivalent(r0, r1, _cfg):
        return r0.strip().lower() == r1.strip().lower()

    # Monkey‑patch the remote call for offline testing
    globals()["remote_judge_equivalent"] = mock_judge_equivalent

    def test_partition():
        solution_str = [
            "Paris is the capital of France.",
            "The capital of France is Paris.",
            "France's capital is Paris.",
            "Why do programmers prefer dark mode? Because light attracts bugs!",
            "Why do programmers prefer dark mode? Because light attracts bugs.",
            "This joke only works on my machine.",
        ]
        uid = ["uid1", "uid1", "uid1", "uid2", "uid2", "uid2"]
        prompts = [
            "What is the capital of France?", "", "", "", "", "",
        ]
        rewards = partition(solution_str=solution_str, uid=uid, prompts=prompts)
        for r, rew in zip(solution_str, rewards):
            print(f"{rew:.2f} → {r}")

    test_partition()
