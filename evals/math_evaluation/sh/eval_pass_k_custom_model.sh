PROMPT_TYPE="qwen-boxed"
MODEL_NAME_OR_PATH=$1

OUTPUT_DIR=${MODEL_NAME_OR_PATH}/eval_output_ob
SPLIT="test"
NUM_TEST_SAMPLE=-1
SEED=0
# English open datasets
#DATA_NAME="math,olympiadbench"
#CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
#python3 -u math_eval.py \
#    --model_name_or_path ${MODEL_NAME_OR_PATH} \
#    --data_name ${DATA_NAME} \
#    --output_dir ${OUTPUT_DIR} \
#    --split ${SPLIT} \
#    --prompt_type ${PROMPT_TYPE} \
#    --num_test_sample ${NUM_TEST_SAMPLE} \
#    --seed ${SEED} \
#    --temperature 0.6 \
#    --n_sampling 128 \
#    --top_p 0.95 \
#    --start 0 \
#    --end -1 \
#    --use_vllm \
#    --save_outputs \
#    --overwrite &
    
# English competition datasets
DATA_NAME="olympiadbench,aime24,aime25"
CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
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
