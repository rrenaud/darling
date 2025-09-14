PROMPT_TYPE="qwen-boxed"
MODEL_NAME_OR_PATH=$1

echo "Running evaluation with prompt type: $PROMPT_TYPE and model: $MODEL_NAME_OR_PATH"
export CUDA_VISIBLE_DEVICES="0"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

