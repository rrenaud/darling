import random
import os
import argparse
import time
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Queue
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions

import numpy as np

# ---------------------------------------------------------------------------
# HELPERS -------------------------------------------------------------------
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute pass@k for a single task.

    :param n: total samples generated for this task
    :param c: number of correct samples among them
    :param k: desired k (k ≤ n)
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def gpus_per_worker(model_name: str) -> int:
    """
    Decide how many GPUs a single worker should reserve.
      • 1 GPU for 4B / 8B models
      • 2 GPUs for 14B models
    """
    name = model_name.lower()
    if "14b" in name or '32b' in name:
        return 4
    return 1  # covers 4B, 8B, and anything else

# ---------------------------------------------------------------------------
# ARGUMENT PARSING ----------------------------------------------------------
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--top_k", default=20, type=int)
    parser.add_argument("--max_tokens_per_call", default=8192, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    parser.add_argument(
        "--aggregate", 
        action="store_true",
        help="Aggregate solutions and run a second evaluation"
    )
    
    args = parser.parse_args()
    args.top_p = 1 if args.temperature == 0 else args.top_p  # top_p must be 1 when using greedy sampling (vllm)
    return args

# ---------------------------------------------------------------------------
# DATA PREPARATION ----------------------------------------------------------
# ---------------------------------------------------------------------------

def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
            )

    # deduplicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file

# ---------------------------------------------------------------------------
# WORKER --------------------------------------------------------------------
# ---------------------------------------------------------------------------

def worker_fn(gpu_ids_str, gpu_group_size, worker_args,
              input_queue, output_queue, init_queue):
    """
    gpu_ids_str     – comma-separated list, e.g. "0" or "0,1"
    gpu_group_size  – number of GPUs reserved for this worker
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str

    args, model_name_or_path, apply_chat_template, prompt_type, \
    temperature, top_p, max_tokens_per_call = worker_args

    # Wait for signal to initialize
    init_signal = init_queue.get()
    if init_signal != "init":
        return

    label = f"[GPUs {gpu_ids_str}]"
    try:
        print(f"{label} Loading model …")
        if args.use_vllm:
            llm = LLM(
                model=model_name_or_path,
                tensor_parallel_size=gpu_group_size,      # ADAPTIVE
                pipeline_parallel_size=1,
                gpu_memory_utilization=0.8,
                trust_remote_code=True,
            )
            tokenizer = None
            if apply_chat_template:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name_or_path, trust_remote_code=True
                )
        else:
            llm, tokenizer = load_hf_lm_and_tokenizer(
                model_name_or_path=model_name_or_path,
                load_in_half=True,
                use_fast_tokenizer=True,
                use_safetensors=args.use_safetensors,
            )

        print(f"{label} Model loaded successfully")
        # Signal that initialization is complete
        output_queue.put(("init_done", gpu_ids_str))

    except Exception as e:
        print(f"{label} Failed to load model: {e}")
        output_queue.put(("init_failed", gpu_ids_str))
        return

    # -----------------------------------------------------------------------
    # MAIN LOOP -------------------------------------------------------------
    # -----------------------------------------------------------------------
    while True:
        task = input_queue.get()
        if task is None:  # Sentinel value to stop worker
            break

        task_id, prompts, stop_words = task

        try:
            # Generate outputs
            if args.use_vllm:
                outputs = llm.generate(
                    prompts,
                    SamplingParams(
                        temperature=temperature,
                        top_p=top_p,
                        top_k=args.top_k,
                        max_tokens=max_tokens_per_call,
                        n=1,
                        stop=stop_words,
                        stop_token_ids=(
                            [151645, 151643]
                            if "qwen2" in model_name_or_path.lower()
                            else None
                        ),
                    ),
                )
                outputs = sorted(outputs, key=lambda x: int(x.request_id))
                outputs = [output.outputs[0].text for output in outputs]
            else:
                outputs = generate_completions(
                    model=llm,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    max_new_tokens=max_tokens_per_call,
                    batch_size=32,
                    stop_id_sequences=stop_words,
                )

            output_queue.put((task_id, outputs))
        except Exception as e:
            print(f"{label} Generation failed: {e}")
            output_queue.put((task_id, None))

# ---------------------------------------------------------------------------
# SETUP & DRIVER ------------------------------------------------------------
# ---------------------------------------------------------------------------

def setup(args):
    # All visible GPUs
    all_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",") if "CUDA_VISIBLE_DEVICES" in os.environ else []
    if not all_gpus or all_gpus == ['']:
        raise RuntimeError("No GPUs visible via CUDA_VISIBLE_DEVICES")

    group_size = gpus_per_worker(args.model_name_or_path)
    gpu_groups = [all_gpus[i:i + group_size] for i in range(0, len(all_gpus), group_size)]

    # Final group must have required size
    if len(gpu_groups[-1]) < group_size:
        raise RuntimeError(
            f"Model requires {group_size} GPU(s) per worker but only "
            f"{len(gpu_groups[-1])} GPU(s) remain."
        )

    num_workers = len(gpu_groups)
    print(f"Detected {len(all_gpus)} physical GPU(s) → spawning {num_workers} worker(s) "
          f"using {group_size} GPU(s) each.")

    # -----------------------------------------------------------------------
    # Prepare shared queues
    # -----------------------------------------------------------------------
    input_queues = [Queue() for _ in range(num_workers)]
    init_queues  = [Queue() for _ in range(num_workers)]
    output_queue = Queue()

    # Worker-specific static args tuple
    worker_args = (
        args,
        args.model_name_or_path,
        args.apply_chat_template,
        args.prompt_type,
        args.temperature,
        args.top_p,
        args.max_tokens_per_call
    )

    # -----------------------------------------------------------------------
    # Spawn workers
    # -----------------------------------------------------------------------
    workers = []
    for i, gpu_grp in enumerate(gpu_groups):
        p = mp.Process(
            target=worker_fn,
            args=(
                ",".join(gpu_grp),          # "0" or "0,1"
                len(gpu_grp),               # group size
                worker_args,
                input_queues[i],
                output_queue,
                init_queues[i],
            ),
        )
        p.start()
        workers.append(p)

    # -----------------------------------------------------------------------
    # Sequential initialization to avoid race in compilation
    # -----------------------------------------------------------------------
    print("Initializing workers sequentially …")
    for i in range(num_workers):
        print(f"→ Worker-{i} (GPUs {','.join(gpu_groups[i])}) initializing")
        init_queues[i].put("init")
        while True:
            status, label = output_queue.get()
            if status == "init_done":
                print(f"✓ Worker-{i} ({label}) ready")
                break
            elif status == "init_failed":
                print(f"✗ Worker-{i} ({label}) failed to start")
                # Clean up and exit
                for q in input_queues:
                    q.put(None)
                for p in workers:
                    p.terminate()
                raise RuntimeError(f"Failed to initialize worker {label}")
        if i < num_workers - 1:
            time.sleep(2)

    print("All workers initialized successfully!")

    # -----------------------------------------------------------------------
    # Run over each dataset
    # -----------------------------------------------------------------------
    data_list = args.data_names.split(",")
    results = []

    for data_name in data_list:
        result = main_parallel(
            input_queues, output_queue, num_workers, data_name, args
        )
        results.append(result)

    # Stop workers
    for q in input_queues:
        q.put(None)
    for p in workers:
        p.join()

    # -----------------------------------------------------------------------
    # Aggregate & log results
    # -----------------------------------------------------------------------
    #data_list.append("avg")
    #results.append({
    #    "acc": sum(r["acc"] for r in results) / len(results),
    #})

    #pad = max(len(name) for name in data_list)
    #print("\t".join(name.ljust(pad) for name in data_list))
    #print("\t".join(f"{r['acc']:.1f}".ljust(pad) for r in results))

# ---------------------------------------------------------------------------
# MAIN PARALLEL LOGIC -------------------------------------------------------
# ---------------------------------------------------------------------------

def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True

def main_parallel(input_queues, output_queue, num_workers, data_name, args):
    """Main function modified for parallel execution"""
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # Init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    # Build sample list
    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # Parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)

        if idx == args.start:
            print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        # add remaining fields
        for key in [
            "level", "type", "unit", "solution_type", "choices", "solution",
            "ques_type", "ans_type", "answer_type", "dataset", "subfield",
            "filed", "theorem", "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # ----------------------------------------------------------------------------
    # DUPLICATE PROMPTS AND DISTRIBUTE ACROSS WORKERS
    # ----------------------------------------------------------------------------
    base_prompts = [s["prompt"] for s in samples]

    if args.apply_chat_template:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True
        )
        base_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in base_prompts
        ]

    # split for each worker
    samples_per_worker = [[] for _ in range(num_workers)]
    for i, sample in enumerate(samples):
        samples_per_worker[i % num_workers].append((i, sample))

    # build duplicated prompts
    gpu_prompts = []
    sample_mapping = {}  # global_prompt_idx -> (sample_idx, dup_idx)
    global_idx = 0
    for worker_id, worker_samples in enumerate(samples_per_worker):
        worker_prompt_list = []
        for sample_idx, sample in worker_samples:
            for dup_idx in range(args.n_sampling):
                worker_prompt_list.append(base_prompts[sample_idx])
                sample_mapping[global_idx] = (sample_idx, dup_idx)
                global_idx += 1
        gpu_prompts.append(worker_prompt_list)

    # ----------------------------------------------------------------------------
    # INFERENCE LOOP
    # ----------------------------------------------------------------------------
    max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")

    # tracking structures sized by num_workers
    remain_prompts_per_worker = [[(i, p) for i, p in enumerate(prompts)]
                                 for prompts in gpu_prompts]
    end_prompts_per_worker    = [[] for _ in range(num_workers)]

    start_time = time.time()

    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)

        # submit tasks
        active_tasks = {}
        for w in range(num_workers):
            if remain_prompts_per_worker[w]:
                prompts = [item[1] for item in remain_prompts_per_worker[w]]
                task_id = f"{epoch}_{w}"
                input_queues[w].put((task_id, prompts, stop_words))
                active_tasks[task_id] = w

        # collect
        results_by_worker = {}
        for _ in range(len(active_tasks)):
            task_id, outputs = output_queue.get()
            w = active_tasks[task_id]
            results_by_worker[w] = outputs

        # process outputs
        for w in range(num_workers):
            if w not in results_by_worker:
                continue

            outputs = results_by_worker[w]
            current_prompts = remain_prompts_per_worker[w]

            new_remain = []
            remain_codes = []

            for (i, query), output in zip(current_prompts, outputs):
                output = output.rstrip()
                query += output

                if args.prompt_type == "pal":
                    new_remain.append((i, query))
                    if "```python" in output:
                        output = extract_program(query)
                    remain_codes.append(output)
                elif args.prompt_type == "cot":
                    end_prompts_per_worker[w].append((i, query))
                elif "boxed" not in output and output.endswith("```"):
                    program = extract_program(query)
                    new_remain.append((i, query))
                    remain_codes.append(program)
                else:
                    end_prompts_per_worker[w].append((i, query))

            # execute remain codes
            if remain_codes:
                remain_results = executor.batch_apply(remain_codes)
                for k in range(len(new_remain)):
                    i, query = new_remain[k]
                    res, report = remain_results[k]
                    exec_result = res if res else report
                    if "pal" in args.prompt_type:
                        exec_result = "\\boxed{" + exec_result + "}"
                    exec_result = f"\n```output\n{exec_result}\n```\n"
                    query += exec_result

                    if epoch == max_func_call - 1:
                        query += "\nReach max function call limit."
                    new_remain[k] = (i, query)

            remain_prompts_per_worker[w] = new_remain

    # ------------------------------------------------------------------------
    # COLLECT ALL END PROMPTS
    # ------------------------------------------------------------------------
    for w in range(num_workers):
        end_prompts_per_worker[w].extend(remain_prompts_per_worker[w])

    all_end_prompts = []
    for w, end_prompts in enumerate(end_prompts_per_worker):
        for local_idx, prompt in end_prompts:
            global_idx = sum(len(gpu_prompts[j]) for j in range(w)) + local_idx
            all_end_prompts.append((global_idx, prompt))
    all_end_prompts = sorted(all_end_prompts, key=lambda x: x[0])

    # extract codes
    codes = []
    all_input_prompts = [p for worker_list in gpu_prompts for p in worker_list]
    for i in range(len(all_input_prompts)):
        _, end_prompt = all_end_prompts[i]
        code = end_prompt.split(all_input_prompts[i])[-1].strip()
        for sw in stop_words:
            if sw in code:
                code = code.split(sw)[0].strip()
        codes.append(code)

    # execute predictions
    results = [run_execute(executor, code, args.prompt_type, data_name) for code in codes]
    time_use = time.time() - start_time

    # group predictions by sample
    sample_results = {i: {"codes": [], "preds": [], "reports": []} for i in range(len(samples))}
    for global_idx, (code, result) in enumerate(zip(codes, results)):
        sample_idx, dup_idx = sample_mapping[global_idx]
        pred, report = result
        sample_results[sample_idx]["codes"].append(code)
        sample_results[sample_idx]["preds"].append(pred)
        sample_results[sample_idx]["reports"].append(report)

    # build final samples
    all_samples = []
    for i, sample in enumerate(samples):
        codes_i   = sample_results[i]["codes"]
        preds_i   = sample_results[i]["preds"]
        reports_i = sample_results[i]["reports"]

        # clean multiple-choice predictions
        for j in range(len(preds_i)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds_i[j] not in ["A", "B", "C", "D", "E"]:
                preds_i[j] = choice_answer_clean(codes_i[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds_i[j]):
                preds_i[j] = "".join([c for c in preds_i[j] if c in ["A", "B", "C", "D", "E"]])

        sample.pop("prompt")
        sample.update({"code": codes_i, "pred": preds_i, "report": reports_i})
        all_samples.append(sample)

    # add processed samples back
    all_samples.extend(processed_samples)

    # evaluate
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )

    # save outputs
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)

    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minite"] = f"{int(time_use // 60)}:{int(time_use % 60):02d}"

    """
    if args.n_sampling > 1 and (args.n_sampling & (args.n_sampling - 1) == 0):
        k_values = [2 ** i for i in range(int(np.log2(args.n_sampling)) + 1)]

        passk_totals = {k: [] for k in k_values}
        for sample in all_samples:
            n = len(sample["pred"])
            c = sum(p == sample["gt"] for p in sample["pred"])
            for k in k_values:
                passk_totals[k].append(pass_at_k(n, c, k))

        passk_avg = {k: float(np.mean(passk_totals[k])) for k in k_values}

        print(" | ".join(f"pass@{k}: {v * 100:.2f}%" for k, v in passk_avg.items()))
        result_json.update({f"pass@{k}": v for k, v in passk_avg.items()})
    """

    if args.n_sampling > 1:
        # For tracking standard metrics
        pass1_values = []      # pass@1 (individual sample accuracy)
        valid_samples = 0
        
        # Bootstrap parameters
        k = 8  # For pass@16 and maj@16
        num_bootstrap_samples = 30  # Number of bootstrap iterations
        bootstrap_passk_values = []  # Store pass@k results for all bootstraps
        bootstrap_majk_values = []   # Store maj@k results for all bootstraps
        
        for sample in all_samples:
            preds = sample["pred"]
            if not preds or len(preds) < args.n_sampling:  # Skip invalid samples
                continue
            
            valid_samples += 1
            gt = sample["gt"]
            
            # Pass@1: Calculate individual sample accuracy (no bootstrapping)
            correct_preds = [p == gt for p in preds]
            pass1_values.extend(correct_preds)
            
            # Perform bootstrap sampling
            for _ in range(num_bootstrap_samples):
                # Sample k predictions with replacement
                bootstrap_indices = np.random.choice(len(preds), size=k, replace=True)
                bootstrap_preds = [preds[i] for i in bootstrap_indices]
                
                # pass@k: 1 if any prediction is correct
                has_correct = any(bootstrap_preds[i] == gt for i in range(k))
                bootstrap_passk_values.append(1.0 if has_correct else 0.0)
                
                # maj@k: majority voting
                from collections import Counter
                pred_counts = Counter(bootstrap_preds)
                majority_pred = pred_counts.most_common(1)[0][0]
                bootstrap_majk_values.append(1.0 if majority_pred == gt else 0.0)
        
        if valid_samples > 0:
            # Calculate averages
            pass1_avg = float(np.mean(pass1_values))
            bootstrap_passk_avg = float(np.mean(bootstrap_passk_values))
            bootstrap_majk_avg = float(np.mean(bootstrap_majk_values))
            
            print(f"pass@1: {pass1_avg * 100:.2f}%")
            print(f"pass@{k} (bootstrapped): {bootstrap_passk_avg * 100:.2f}%")
            print(f"maj@{k} (bootstrapped): {bootstrap_majk_avg * 100:.2f}%")
            
            result_json["pass@1"] = pass1_avg
            result_json[f"pass@{k}_bootstrapped"] = bootstrap_passk_avg
            result_json[f"maj@{k}_bootstrapped"] = bootstrap_majk_avg
        
        with open(
            out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w"
        ) as f:
            json.dump(result_json, f, indent=4)
    
    
    return result_json


# ---------------------------------------------------------------------------
# ENTRYPOINT ----------------------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    mp.set_start_method('spawn', force=True)  # Required for CUDA multiprocessing
    setup(args)
