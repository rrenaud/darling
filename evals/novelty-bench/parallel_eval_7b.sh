MODEL_NAME=$(echo $1 | awk -F'/' '{print $NF}')
# if "global_step" is in the name, append the last part before the last "/" to the model name
if [[ $MODEL_NAME == *"global_step"* ]]; 
then
	MODEL_SETUP=$(echo $1 | awk -F'/' '{print $(NF-1)}')
	MODEL_NAME="${MODEL_SETUP}_${MODEL_NAME}"
fi


TEMPERATURE=1.0

echo "MODEL_NAME: $MODEL_NAME"
# if $2 is not empty, set it as the temperature
if [ -n "$2" ]; then
    TEMPERATURE=$2
fi
echo "TEMPERATURE: $TEMPERATURE"
set -x

EVAL_DIR=$SCRATCH_DIR/novelty_bench_generations/${MODEL_NAME}_n10_parallel_t${TEMPERATURE}
if ! test -f "${EVAL_DIR}/generations.jsonl"; then
	echo "Creating evaluation directory: ${EVAL_DIR}"
	mkdir -p ${EVAL_DIR}
	vllm serve $1 --served-model-name ${MODEL_NAME} --max-model-len 1024 --tensor-parallel 2 --tokenizer /datasets/pretrained-llms/Llama-3.1-8B-Instruct &
	sleep 30
	export PYTHONPATH=$(pwd):$PYTHONPATH
	python3 src/inference.py \
		--mode vllm \
		--model ${MODEL_NAME} \
		--temperature ${TEMPERATURE} \
		--eval-dir ${EVAL_DIR} \
		--data both \
		--sampling regenerate \
		--num-generations 10 
	pkill -f 'vllm serve'
else
	echo "Directory ${EVAL_DIR} already exists. Skipping creation."
fi
# sleep for 30 seconds to ensure the vllm server is up and running
python3 /home/$USER/darling/evals/novelty-bench/src/partition.py --eval-dir ${EVAL_DIR} --alg classifier 
python3 /home/$USER/darling/evals/novelty-bench/src/score.py --eval-dir ${EVAL_DIR}
python3 /home/$USER/darling/evals/novelty-bench/src/summarize.py --eval-dir ${EVAL_DIR}

# print the results
cat ${EVAL_DIR}/summary.json
