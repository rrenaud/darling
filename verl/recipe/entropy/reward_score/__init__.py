import traceback

from . import entropy_math


def _default_compute_score(
    data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None
):
    try:
        res = entropy_math.compute_score(solution_str, str(ground_truth))
        # print(f"data_source: {data_source}")
        # raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

        if isinstance(res, dict):
            return res
        elif isinstance(res, int | float | bool):
            return float(res)
        else:
            return float(res[0])
    except Exception as e:
        print(f"[ERROR] Error in process_completion for task : {str(e)}")
        traceback.print_exc()  # 打印完整堆栈
        raise  # 重新抛出异常以便上层捕获
