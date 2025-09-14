import torch
import functools
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import asyncio
from collections import defaultdict


# ngrams
async def process_uid_group(uid, resp_list, uid_to_original_index, n):
    """Process a single uid group and return rewards for its responses."""
    # First, collect all n-grams for each response
    response_ngrams = []
    
    for resp in resp_list:
        ngrams = set(tuple(resp[i:i+n]) for i in range(len(resp) - n + 1))
        response_ngrams.append(ngrams)
    
    group_rewards = []
    
    # For each response, find n-grams that are unique to this response
    for i, ngrams in enumerate(response_ngrams):
        # Create a set of all n-grams in all other responses
        other_ngrams = set()
        for j, other_ngrams_set in enumerate(response_ngrams):
            if i != j:  # Skip the current response
                other_ngrams.update(other_ngrams_set)
        
        # Calculate unique n-grams (present in this response but no others)
        unique_ngrams = ngrams - other_ngrams
        
        # Normalize by the total number of n-grams in this response
        # Handle edge case where response is too short to form n-grams
        total_ngrams = max(1, len(resp_list[i]) - n + 1) if len(resp_list[i]) >= n else 1
        reward = len(unique_ngrams) / total_ngrams
        
        original_index = uid_to_original_index[uid][i]
        group_rewards.append((original_index, reward))
    
    return group_rewards

async def ngram_async(**kwargs):
    solution_str = kwargs.get("solution_str", [])
    uid = kwargs.get("uid", None)
    if uid is None:
        raise ValueError("uid is required for ngram reward function")
    n = kwargs.get("n", 4)

    responses = [s.split() for s in solution_str]
    batch_size = len(responses)
    
    # create a dictionary mapping uid to a list of responses
    uid_to_responses = {}
    uid_to_original_index = {}
    for i in range(batch_size):
        if uid[i] not in uid_to_responses:
            uid_to_responses[uid[i]] = []
            uid_to_original_index[uid[i]] = []
        uid_to_responses[uid[i]].append(responses[i])
        uid_to_original_index[uid[i]].append(i)
    
    # Process each uid group in parallel
    tasks = []
    for uid, resp_list in uid_to_responses.items():
        task = process_uid_group(uid, resp_list, uid_to_original_index, n)
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    print(f"Results: {results}")
    
    # Reconstruct rewards in original order
    rewards = [0.0] * batch_size
    for group_rewards in results:
        for original_index, reward in group_rewards:
            rewards[original_index] = reward
    
    return rewards

# Synchronous wrapper for backward compatibility
def ngram(**kwargs):
    """Synchronous wrapper that runs the async version."""
    return asyncio.run(ngram_async(**kwargs))

# Partitions

@functools.cache
def load_deberta_tokenizer_and_model():

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    model = AutoModelForSequenceClassification.from_pretrained(
        "yimingzhang/deberta-v3-large-generation-similarity"
    ).to(DEVICE)
    model.eval()
    return tokenizer, model

@functools.cache
def load_modernbert_tokenizer_and_model():

    DEVICE = "cpu" if torch.cuda.is_available() else "cuda"
    tokenizer = AutoTokenizer.from_pretrained(
        local_dir='/checkpoint/ram/tianjian/modernbert-similarity-classifier',
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        local_dir='/checkpoint/ram/tianjian/modernbert-similarity-classifier',
    ).to(DEVICE)
    model.eval()
    return tokenizer, model

@torch.inference_mode()
async def classifier_score(prompt: str, s1: str, s2: str, model: str):
    if model == 'modernbert':
        tokenizer, model = load_modernbert_tokenizer_and_model()
    elif model == 'deberta':
        tokenizer, model = load_deberta_tokenizer_and_model()
    else:
        raise ValueError("Unsupported model type. Use 'modernbert' or 'deberta'.")
    
    input_ids = [tokenizer.cls_token_id]
    for s in [s1, s2]:
        input_ids.extend(
            tokenizer.encode(
                s,
                truncation=True,
                max_length=128,
                add_special_tokens=False,
            )
        )
        input_ids.append(tokenizer.sep_token_id)
        prompt_len = input_ids.index(tokenizer.sep_token_id) + 1
    token_type_ids = [0] * prompt_len + [1] * (len(input_ids) - prompt_len)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    iids = torch.tensor(input_ids, device=DEVICE, dtype=torch.int64)
    tids = torch.tensor(token_type_ids, device=DEVICE, dtype=torch.int64)

    outputs = model(input_ids=iids.unsqueeze(0), token_type_ids=tids.unsqueeze(0))
    score = outputs["logits"].softmax(-1)[0, 1]
    return score.cpu().item()


def maybe_test_equality(response_0: str, response_1: str) -> bool | None:
    unigram_0 = response_0.strip().lower().split()
    unigram_1 = response_1.strip().lower().split()
    max_len = max(len(unigram_0), len(unigram_1))
    if max_len <= 5:
        common_unigrams = set(unigram_0) & set(unigram_1)
        return len(common_unigrams) * 2 >= max_len

    return None


async def equivalence_check_classifier(
    prompt: str,
    response_0: str,
    response_1: str,
    model: str,
) -> bool:
    equality = maybe_test_equality(response_0, response_1)
    if equality is not None:
        return equality
    score = await classifier_score(prompt, response_0, response_1, model=model)
    #print(f"Classifier score for '{response_0}' and '{response_1}': {score}")
    if model == 'modernbert':
        return score > 0.102
    elif model == 'deberta':
        return score > 0.102
    else:
        raise ValueError("Unsupported model type. Use 'modernbert' or 'deberta'.")

import asyncio

async def process_uid_partition(user_id, responses, indices, prompt, model):
    """Process partitioning for a single uid group, concurrent pairwise checks."""
    n = len(responses)

    # Union-Find data structure for efficient partitioning
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Step 1: Prepare all pairs to check
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Step 2: Run all checks concurrently
    async def check_pair(i, j):
        result = await equivalence_check_classifier(prompt, responses[i], responses[j], model)
        return (i, j, result)

    tasks = [check_pair(i, j) for i, j in pairs]
    results = await asyncio.gather(*tasks)

    # Step 3: Union equivalent pairs
    for i, j, equivalent in results:
        if equivalent:
            union(i, j)

    # Step 4: Group by partition
    partition_map = {}
    for i in range(n):
        root = find(i)
        if root not in partition_map:
            partition_map[root] = []
        partition_map[root].append(indices[i])

    # Return uid and its partitions
    return user_id, list(partition_map.values())

async def partition_async(solution_str, uid, prompts, model):
    """
    Partition responses into groups of semantically equivalent responses.
    Process each uid group in parallel.
    
    Args:
        solution_str: List of response strings
        uid: List of user IDs corresponding to each response
        prompts: List of prompts corresponding to each response
    
    Returns:
        Dictionary mapping uid to list of partition groups, where each group 
        contains indices of semantically equivalent responses
    """
    # Group responses by uid
    uid_to_responses = {}
    uid_to_original_index = {}
    uid_to_prompt = {}
    
    for i, (response, user_id, prompt) in enumerate(zip(solution_str, uid, prompts)):
        if user_id not in uid_to_responses:
            uid_to_responses[user_id] = []
            uid_to_original_index[user_id] = []
            uid_to_prompt[user_id] = prompt
        uid_to_responses[user_id].append(response)
        uid_to_original_index[user_id].append(i)
    
    # Create tasks for parallel processing
    tasks = []
    for user_id, responses in uid_to_responses.items():
        indices = uid_to_original_index[user_id]
        prompt = uid_to_prompt[user_id]
        task = asyncio.create_task(process_uid_partition(user_id, responses, indices, prompt, model))
        tasks.append(task)
    
    # Execute all tasks in parallel
    results = await asyncio.gather(*tasks)
    
    # Convert results to dictionary
    uid_to_partitions = {user_id: partitions for user_id, partitions in results}
    
    return uid_to_partitions

# Synchronous wrapper for backward compatibility
def partition(**kwargs):
    """
    Partition responses and return distinctness rewards.
    
    Returns:
        List of rewards where each reward is the number of responses 
        the response is distinct from within its uid group
    """
    solution_str = kwargs.get("solution_str", [])  
    uid = kwargs.get("uid", None)
    prompts = kwargs.get("prompts", [])
    model='deberta' # Default model, can be overridden
    if uid is None:
        raise ValueError("uid is required for partition reward function")
    
    # Get partitions
    uid_to_partitions = asyncio.run(partition_async(solution_str, uid, prompts, model=model))
    
    # Group responses by uid to get total count per uid
    uid_to_indices = {}
    for i, user_id in enumerate(uid):
        if user_id not in uid_to_indices:
            uid_to_indices[user_id] = []
        uid_to_indices[user_id].append(i)
    
    # Initialize rewards
    rewards = [0] * len(solution_str)
    
    # Calculate rewards for each uid group
    for user_id, partitions in uid_to_partitions.items():
        total_responses = len(uid_to_indices[user_id])
        
        # For each partition group
        for group in partitions:
            group_size = len(group)
            # Each response in this group is distinct from all responses NOT in this group
            distinct_count = total_responses - group_size
            
            # Assign reward to each response in the group
            for idx in group:
                rewards[idx] = distinct_count 
    
    return [reward / len(solution_str) for reward in rewards] 

def partition_modernbert(**kwargs):
   
    solution_str = kwargs.get("solution_str", [])  
    uid = kwargs.get("uid", None)
    prompts = kwargs.get("prompts", [])
    model='modernbert' # Default model, can be overridden
    if uid is None:
        raise ValueError("uid is required for partition reward function")
    
    # Get partitions
    uid_to_partitions = asyncio.run(partition_async(solution_str, uid, prompts, model=model))
    
    # Group responses by uid to get total count per uid
    uid_to_indices = {}
    for i, user_id in enumerate(uid):
        if user_id not in uid_to_indices:
            uid_to_indices[user_id] = []
        uid_to_indices[user_id].append(i)
    
    # Initialize rewards
    rewards = [0] * len(solution_str)
    
    # Calculate rewards for each uid group
    for user_id, partitions in uid_to_partitions.items():
        total_responses = len(uid_to_indices[user_id])
        
        # For each partition group
        for group in partitions:
            group_size = len(group)
            # Each response in this group is distinct from all responses NOT in this group
            distinct_count = total_responses - group_size
            
            # Assign reward to each response in the group
            for idx in group:
                rewards[idx] = distinct_count 
    # normalize by the number of total responses
    return [reward / len(solution_str) for reward in rewards]
    
async def process_unlikelihood_group(uid, responses, log_probs, indices):
    """Process a single uid group and calculate unlikelihood rewards."""
    group_size = len(responses)
    if group_size <= 1:
        # For single response in a group, assign neutral reward
        return [(indices[0], 0.5)] if group_size == 1 else []
    
    # Normalize log probabilities by length
    norm_log_probs = []
    for resp, log_prob in zip(responses, log_probs):
        # Get token length for normalization
        length = len(resp.split())
        # Avoid division by zero
        if length > 0:
            norm_log_prob = log_prob / length
        else:
            norm_log_prob = log_prob
        norm_log_probs.append(norm_log_prob)
    
    # Scale rewards within group: lower probability = higher reward
    min_prob = min(norm_log_probs)
    max_prob = max(norm_log_probs)
    
    group_rewards = []
    for i, norm_log_prob in enumerate(norm_log_probs):
        if max_prob == min_prob:
            # All responses have same probability
            reward = 0.5
        else:
            # Invert so lower probability = higher reward
            reward = 1.0 - (norm_log_prob - min_prob) / (max_prob - min_prob)
        group_rewards.append((indices[i], reward))
    
    return group_rewards

async def unlikelihood_async(**kwargs):
    """
    Reward responses with lower probabilities (normalized by length).
    
    Args:
        solution_str: List of response strings
        uid: List of user IDs corresponding to each response
        log_probs: List of log probabilities for each response
    
    Returns:
        List of rewards where lower probability = higher reward
    """
    solution_str = kwargs.get("solution_str", [])
    uid = kwargs.get("uid", None)
    log_probs = kwargs.get("log_probs", None)
    
    if uid is None:
        raise ValueError("uid is required for unlikelihood reward function")
    if log_probs is None:
        raise ValueError("log_probs is required for unlikelihood reward function")
    
    batch_size = len(solution_str)
    
    # Group by uid
    uid_to_responses = {}
    uid_to_log_probs = {}
    uid_to_original_index = {}
    
    for i in range(batch_size):
        if uid[i] not in uid_to_responses:
            uid_to_responses[uid[i]] = []
            uid_to_log_probs[uid[i]] = []
            uid_to_original_index[uid[i]] = []
        uid_to_responses[uid[i]].append(solution_str[i])
        uid_to_log_probs[uid[i]].append(log_probs[i])
        uid_to_original_index[uid[i]].append(i)
    
    # Process each uid group in parallel
    tasks = []
    for uid, responses in uid_to_responses.items():
        task = process_unlikelihood_group(
            uid, 
            responses, 
            uid_to_log_probs[uid], 
            uid_to_original_index[uid]
        )
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    # Reconstruct rewards in original order
    rewards = [0.0] * batch_size
    for group_rewards in results:
        for original_index, reward in group_rewards:
            rewards[original_index] = reward
    
    return rewards

def unlikelihood(**kwargs):
    """
    Synchronous wrapper for unlikelihood_async.
    Rewards responses with lower probabilities.
    """
    return asyncio.run(unlikelihood_async(**kwargs))

def test_ngram():
    print("=== Testing ngram function ===")
    
    # Test data - 2 prompts, each with 3 responses (2 similar, 1 distinct)
    solution_str = [
        # Prompt 1 responses
        "the cat sat on the mat",             # Similar response 1
        "the cat sat on the carpet",          # Similar response 2 
        "elephants are large gray animals",   # Distinct response
        
        # Prompt 2 responses
        "python is a programming language",   # Similar response 1
        "python is a coding language",        # Similar response 2
        "dolphins swim in the ocean"          # Distinct response
    ]
    
    # User IDs - 2 prompts, each with 3 responses
    uid = ["prompt1", "prompt1", "prompt1", "prompt2", "prompt2", "prompt2"]
    
    # Call the function with bigrams for more meaningful differences
    print("Running ngram function with n=2...")
    rewards = ngram(solution_str=solution_str, uid=uid, n=2)
    
    print(f"\nReward scores: {rewards}")
    
    # Check first prompt group (index 0-2)
    print("\nChecking first prompt group (prompt1):")
    print(f"Similar response 1: {rewards[0]:.4f}")
    print(f"Similar response 2: {rewards[1]:.4f}")
    print(f"Distinct response: {rewards[2]:.4f}")
    
    if rewards[2] > rewards[0] and rewards[2] > rewards[1]:
        print("✓ PASS: Distinct response has higher reward than similar responses")
    else:
        print("✗ FAIL: Distinct response should have higher reward")
        
    if abs(rewards[0] - rewards[1]) < 0.2:
        print("✓ PASS: Similar responses have comparable rewards")
    else:
        print("✗ FAIL: Similar responses should have comparable rewards")
    
    # Check second prompt group (index 3-5)
    print("\nChecking second prompt group (prompt2):")
    print(f"Similar response 1: {rewards[3]:.4f}")
    print(f"Similar response 2: {rewards[4]:.4f}")
    print(f"Distinct response: {rewards[5]:.4f}")
    
    if rewards[5] > rewards[3] and rewards[5] > rewards[4]:
        print("✓ PASS: Distinct response has higher reward than similar responses")
    else:
        print("✗ FAIL: Distinct response should have higher reward")
        
    if abs(rewards[3] - rewards[4]) < 0.2:
        print("✓ PASS: Similar responses have comparable rewards")
    else:
        print("✗ FAIL: Similar responses should have comparable rewards")
    
    # Check valid range for rewards
    print("\nChecking reward ranges:")
    valid_range = all(0 <= r <= 1 for r in rewards)
    if valid_range:
        print("✓ PASS: All rewards are in the valid range [0, 1]")
    else:
        print("✗ FAIL: Some rewards are outside the valid range [0, 1]")
    
    # Test error case
    print("\nChecking error handling:")
    try:
        ngram(solution_str=["test"])
        print("✗ FAIL: Function should raise error when uid is missing")
    except ValueError as e:
        print(f"✓ PASS: Function correctly raised error: {e}")

def test_empty_responses():
    print("\n=== Testing empty responses ===")
    solution_str = ["", "", ""]
    uid = ["prompt1", "prompt1", "prompt1"]
    
    rewards = ngram(solution_str=solution_str, uid=uid)
    print(f"Rewards for empty responses: {rewards}")
    
    if rewards == [0.0, 0.0, 0.0]:
        print("✓ PASS: Empty responses all have zero rewards")
    else:
        print("✗ FAIL: Empty responses should have zero rewards")

if __name__ == "__main__":
    test_ngram()
