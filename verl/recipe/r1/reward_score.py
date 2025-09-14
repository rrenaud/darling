def reward_func(data_source, solution_str, ground_truth, extra_info=None):
    if data_source in ["Maxwell-Jia/AIME_2024", "opencompass/cnmo2024_en", "opencompass/cnmo2024_zh"]:
        from recipe.r1.tasks import math

        return math.compute_score(solution_str, ground_truth)
    elif data_source == "Idavidrein/gpqa":
        from recipe.r1.tasks import gpqa

        return gpqa.compute_score(solution_str, ground_truth)
    elif data_source in ["livecodebench/code_generation_lite", "livecodebench/code_generation"]:
        from recipe.r1.tasks import livecodebench

        return livecodebench.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError
