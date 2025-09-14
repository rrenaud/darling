MODEL_PATH=$1
MODEL_NAME=$(basename $1)

echo "now performing eq bench (creative writing) evaluation for model ${MODEL_NAME}"

vllm serve $1 --served-model-name $(basename $1) --max-model-len 16384 --tensor-parallel 1 --port 8000 --host 0.0.0.0 &

sleep 30

python3 creative_writing_bench.py \
    --test-model $MODEL_NAME \
    --judge-model "anthropic/claude-3.7-sonnet" \
    --runs-file "creative_bench_runs.json" \
    --creative-prompts-file "data/creative_writing_prompts_v3.json" \
    --run-id $MODEL_NAME \
    --threads 500 \
    --verbosity "INFO" \
    --iterations 3
