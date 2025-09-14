def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None, shape=None, **kwargs):
    
    from math_verify import parse, verify
    solution_str = [parse(sol) for sol in solution_str] if isinstance(solution_str, list) else parse(solution_str)
    ground_truth = [parse(gt) for gt in ground_truth] if isinstance(ground_truth, list) else parse(ground_truth)
    results = [verify(gt, sol) for gt, sol in zip(ground_truth, solution_str)]
    # config.reward_model.shape
    if shape == [1, -1]:
        results = [1 if res else -1 for res in results]
    elif shape == [2, 1]:
        results = [2 if res else 1 for res in results]
    else: # the default case: "1,0"
        results = [1 if res else 0 for res in results]
    # when using correct:1 incorrect:-1, diverse and incorrect responses are **discouraged**
    # when using correct:2 incorrect:1, diverse and incorrect responses are **encouraged**
    # default: correct:1 incorrect:0, incorrect responses are given the same reward (0) regardless of diversity

    return results 

    if data_source == "openai/gsm8k":
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
        from . import math

        res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source == "math_dapo" or data_source.startswith("aime"):
        from . import math_dapo
        res = math_dapo.compute_score(solution_str, ground_truth)
    
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(sandbox_fusion_url, concurrent_semaphore, solution_str, ground_truth, continuous=True)
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
