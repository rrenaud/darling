set -e

PROMPT_TYPE="qwen-boxed"
MODEL_NAME_OR_PATH="$1" # e.g. /path/to/model
N_SEEDS=8
SEEDS=(0 1 2 3 4 5 6 7)
OUTPUT_ROOT="${MODEL_NAME_OR_PATH}/math_eval_seeded"
EVAL_SCRIPT="sh/eval.sh"

# 1. Launch 8 parallel runs, each with a unique output dir and seed
PIDS=()
for seed in "${SEEDS[@]}"; do
    (
        export CUDA_VISIBLE_DEVICES="$((seed % 8))"
        OUTPUT_DIR="${OUTPUT_ROOT}/seed_${seed}"
        mkdir -p "$OUTPUT_DIR"
        bash "$EVAL_SCRIPT" "$PROMPT_TYPE" "$MODEL_NAME_OR_PATH" "$OUTPUT_DIR" "$seed"
    ) &
    PIDS+=($!)
done

# 2. Wait for all runs to finish
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo "All $N_SEEDS seeds have finished!"

# 3. Aggregate metrics across all seeds
ALL_BENCHMARKS=("math" "olympiadbench" "aime24" "amc23")
METRICS=("acc" "pass@8")

echo ""
echo "Aggregating results (acc, pass@8) for all benchmarks:"
printf "%-20s %-10s %-10s\n" "Benchmark" "avg@8" "pass@8"
echo "-------------------------------------------------------"

for bench in "${ALL_BENCHMARKS[@]}"; do
    accs=()
    pass8s=()
    for seed in "${SEEDS[@]}"; do
        METRIC_FILE="${OUTPUT_ROOT}/seed_${seed}/${bench}"/*_metrics.json
        if [ -f $METRIC_FILE ]; then
            ACC=$(jq '.acc' "$METRIC_FILE")
            PASS8=$(jq '.["pass@8"] // empty' "$METRIC_FILE")
            accs+=("$ACC")
            pass8s+=("$PASS8")
        else
            echo "Warning: No metric file found for $bench, seed $seed!"
        fi
    done
    # Compute mean
    AVG_ACC=$(printf '%s\n' "${accs[@]}" | awk '{sum+=$1} END {if (NR > 0) printf("%.4f", sum/NR); else print "NaN"}')
    AVG_PASS8=$(printf '%s\n' "${pass8s[@]}" | awk '{sum+=$1} END {if (NR > 0) printf("%.4f", sum/NR); else print "NaN"}')
    printf "%-20s %-10s %-10s\n" "$bench" "$AVG_ACC" "$AVG_PASS8"
done

