PROMPT_TYPE="qwen-boxed"
MODEL_NAME_OR_PATH=$1

OUTPUT_DIR=${MODEL_NAME_OR_PATH}/eval_output_20k
SPLIT="test"
NUM_TEST_SAMPLE=-1
SEED=0

# English open datasets
DATA_NAME="olympiadbench"
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
    --n_sampling 128 \
    --top_p 0.95 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite

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
    --n_sampling 256 \
    --top_p 0.95 \
    --start 0 \
    --use_vllm \
    --end -1 \
    --save_outputs \
    --overwrite
'
