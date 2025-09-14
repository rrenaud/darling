PROMPT_TYPE="qwen-boxed"
MODEL_NAME_OR_PATH=$1

OUTPUT_DIR=${MODEL_NAME_OR_PATH}/eval_output_parallel
SPLIT="test"
NUM_TEST_SAMPLE=-1
SEED=0

# English competition datasets
DATA_NAME="brumo,hmmt,aime25"
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval_parallel.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed ${SEED} \
    --temperature 0.6 \
    --n_sampling 32 \
    --max_tokens_per_call 32768 \
    --top_p 0.95 \
    --top_k 20 \
    --start 0 \
    --use_vllm \
    --end -1 \
    --save_outputs \
    --overwrite 

