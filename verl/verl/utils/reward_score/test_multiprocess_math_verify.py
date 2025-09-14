import time
from datasets import Dataset
from typing import List, Union, Optional, Any, Tuple
import multiprocessing as mp
try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


def _default_compute_score_multiprocess(
    data_source, 
    solution_str: Union[str, List[str]], 
    ground_truth: Union[str, List[str]], 
    num_proc: int = None,
    batch_size: int = 1000,
    extra_info=None, 
    sandbox_fusion_url=None, 
    concurrent_semaphore=None, 
    **kwargs
) -> List[int]:
    """
    Multiprocessing version using HuggingFace datasets.map
    
    Args:
        num_proc: Number of processes (None for auto-detect)
        batch_size: Batch size for processing
    """
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    
    # Normalize inputs to lists
    solution_str = [solution_str] if isinstance(solution_str, str) else solution_str
    ground_truth = [ground_truth] if isinstance(ground_truth, str) else ground_truth
    
    # Format ground truths
    ground_truth = ["\\boxed{" + gt + "}" for gt in ground_truth]
    print(ground_truth)
    
    # Create dataset
    data = {
        'solution': solution_str,
        'ground_truth': ground_truth
    }
    dataset = Dataset.from_dict(data)
    
    # Define mapping function
    def verify_batch(examples):
        results = []
        for sol, gt in zip(examples['solution'], examples['ground_truth']):
            result = verify_func([gt], [sol])
            results.append(1 if result[0] else 0)
        return {'score': results}
    
    # Process with multiprocessing
    if num_proc is None:
        num_proc = min(mp.cpu_count(), len(solution_str))
    
    processed = dataset.map(
        verify_batch,
        batched=True,
        batch_size=32,
        num_proc=num_proc,
        remove_columns=['solution', 'ground_truth']
    )
    
    return processed['score']


def _default_compute_score_original(
    data_source, 
    solution_str: Union[str, List[str]], 
    ground_truth: Union[str, List[str]], 
    extra_info=None, 
    sandbox_fusion_url=None, 
    concurrent_semaphore=None, 
    **kwargs
) -> List[int]:
    """Original single-process version for comparison"""
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    
    solution_str = [solution_str] if isinstance(solution_str, str) else solution_str
    ground_truth = ["\\boxed{" + gt + "}" for gt in ground_truth] if isinstance(ground_truth, list) else "\\boxed{" + ground_truth + "}"
    
    results = [verify_func([gt], [sol]) for gt, sol in zip(ground_truth, solution_str)]
    results = [1 if res[0] else 0 for res in results]
    
    return results


# Test cases
def run_performance_tests():
    """Test cases to observe speedup between single and multiprocess versions"""
    
    # Test case 1: Small dataset (3 items)
    print("Test Case 1: Small dataset (3 items)")
    solution_strs_small = [
        "$\\boxed{5}$",
        "$\\boxed{12}$",
        "$\\boxed{4}$"
    ]
    ground_truths_small = ["5", "12", "4"]
    
    # Test case 2: Medium dataset (30 items)
    print("\nTest Case 2: Medium dataset (30 items)")
    solution_strs_medium = solution_strs_small * 10
    ground_truths_medium = ground_truths_small * 10
    
    # Test case 3: Large dataset (300 items)
    print("\nTest Case 3: Large dataset (300 items)")
    solution_strs_large = solution_strs_small * 100
    ground_truths_large = ground_truths_small * 100
    
    test_cases = [
        ("Small (3 items)", solution_strs_small, ground_truths_small),
        ("Medium (30 items)", solution_strs_medium, ground_truths_medium),
        ("Large (300 items)", solution_strs_large, ground_truths_large)
    ]
    
    for name, solutions, truths in test_cases:
        print(f"\n{'='*50}")
        print(f"Running: {name}")
        print(f"{'='*50}")
        
        # Single process
        start = time.time()
        results_single = _default_compute_score_original(
            None, solutions, truths
        )
        time_single = time.time() - start
        
        # Multiprocess with different process counts
        num_procs = 96
        start = time.time()
        results_multi = _default_compute_score_multiprocess(
            None, solutions, truths, num_proc=num_procs
        )
        time_multi = time.time() - start
        
        # Print results
        print(f"Single process results: {results_single}")
        print(f"Multiprocess results:   {results_multi}")
        
        # Verify they match
        assert results_single == results_multi, "Results don't match!"
        print("✓ Results match!")
        
        print(f"Single process time: {time_single:.3f}s")
        print(f"Multiprocess time ({num_procs} procs): {time_multi:.3f}s")

def run_basic_test():
    """Basic functionality test with the original 3 examples"""
    print("Basic Functionality Test")
    print("-" * 30)
    
    solution_strs = [
        "\\boxed{5}",
        "\\boxed{12}", 
        "\\boxed{4}"
    ]
    ground_truths = ["5", "12", "4"]
    
    # Test single process
    results_single = _default_compute_score_original(None, solution_strs, ground_truths)
    print(f"Single process results: {results_single}")
    
    # Test multiprocess
    results_multi = _default_compute_score_multiprocess(
        None, solution_strs, ground_truths, num_proc=96
    )
    print(f"Multiprocess results:   {results_multi}")
    
    # Verify they match
    assert results_single == results_multi, "Results don't match!"
    print("✓ Results match!")


if __name__ == "__main__":
    # Run basic test first
    #run_basic_test()
    #print("\n" + "="*60 + "\n")
    
    # Run performance tests
    run_performance_tests()