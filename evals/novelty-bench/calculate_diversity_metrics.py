import sys
import json
from collections import Counter

def get_ngrams(tokens, n):
    """
    Generates n-grams from a list of tokens.

    Args:
        tokens (list): A list of strings (words).
        n (int): The size of the n-grams.

    Returns:
        list: A list of n-gram tuples.
    """
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def calculate_inter_generation_diversity(generations, n=4):
    """
    Calculates the average proportion of n-grams in each generation that are
    distinct from the n-grams in all other generations for the same prompt.

    For each generation, it computes:
    (number of n-grams unique to this generation) / (total number of n-grams in this generation)

    It then returns the average of these scores over all generations for the prompt.

    Args:
        generations (list): A list of generated strings for a single prompt.
        n (int, optional): The n-gram size. Defaults to 4.

    Returns:
        float: The average cross-generation diversity score for the prompt.
    """
    if not generations or len(generations) <= 1:
        # If 0 or 1 generation, all n-grams are trivially distinct from "others".
        # The normalized score is 1.0 (or 0.0 if no generations).
        return 1.0 if generations else 0.0

    # Step 1: Get lists and sets of n-grams for each generation
    all_ngrams_lists = []
    for gen in generations:
        tokens = gen.split()
        all_ngrams_lists.append(get_ngrams(tokens, n) if len(tokens) >= n else [])
    
    all_ngram_sets = [set(ngrams) for ngrams in all_ngrams_lists]

    generation_scores = []
    for i in range(len(generations)):
        current_ngrams_list = all_ngrams_lists[i]
        current_ngram_set = all_ngram_sets[i]

        total_ngrams_in_current = len(current_ngrams_list)
        if total_ngrams_in_current == 0:
            generation_scores.append(0.0)
            continue

        # Step 2: Create a combined set of n-grams from all *other* generations
        other_ngram_sets = all_ngram_sets[:i] + all_ngram_sets[i+1:]
        union_of_others = set().union(*other_ngram_sets)

        # Step 3: Find n-grams in the current generation that are not in the others
        distinct_to_current = current_ngram_set.difference(union_of_others)
        num_distinct = len(distinct_to_current)

        # Step 4: Normalize by the total number of n-grams in the current generation
        normalized_score = num_distinct / total_ngrams_in_current
        normalized_score = num_distinct
        generation_scores.append(normalized_score)

    # Step 5: Average the scores for all generations for this prompt
    return sum(generation_scores) / len(generation_scores)

def process_file(file_path, n=4):
    """
    Processes a .jsonl file to calculate the average inter-generation
    n-gram diversity score across all prompts.

    Args:
        file_path (str): The path to the input file.
        n (int, optional): The n-gram size. Defaults to 4.

    Returns:
        float: The average diversity score.
    """
    total_diversity_scores = 0
    num_prompts = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    generations = data.get("generations", [])
                    
                    if not generations:
                        continue

                    # Calculate the diversity score for the list of generations
                    diversity_score = calculate_inter_generation_diversity(generations, n)
                    
                    total_diversity_scores += diversity_score
                    num_prompts += 1

                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line: {line.strip()}")
                except (KeyError, TypeError) as e:
                    print(f"Warning: Skipping malformed data entry. Error: {e}. Line: {line.strip()}")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return 0.0

    if num_prompts == 0:
        print("Warning: No valid prompts with generations found to process.")
        return 0.0

    return total_diversity_scores / num_prompts

def create_sample_data_file(file_path="data.jsonl"):
    """Creates a sample data file for testing."""
    data = [
        {"id": 1, "prompt": "prompt1", "generations": ["the quick brown fox jumps over the lazy dog", "the quick brown fox jumps over the lazy cat"]},
        {"id": 2, "prompt": "prompt2", "generations": ["a stitch in time saves nine", "a stitch in time saves nine indeed"]},
        {"id": 3, "prompt": "prompt3", "generations": ["one two three four five six", "one two three four five six seven"]},
        {"id": 4, "prompt": "prompt4", "generations": ["this is a test this is a test", "that is a test that is a test"]}
    ]
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"Sample data file created at '{file_path}'")


if __name__ == "__main__":
    # --- Create a sample file to run the script ---
    sample_file = sys.argv[1] if len(sys.argv) > 1 else "data.jsonl"

    # --- Main execution ---
    # Set the n-gram size
    for NGRAM_SIZE in [1, 2, 3, 4]:
        print(f"Calculating diversity metrics with n-gram size: {NGRAM_SIZE}")
        #create_sample_data_file(sample_file)
        average_diversity = process_file(sample_file, n=NGRAM_SIZE)
        print(f"Average inter-generation diversity score (n={NGRAM_SIZE}): {average_diversity:.4f}")
        print("-" * 40)
