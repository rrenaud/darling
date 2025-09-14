set -x
# serve 8 instances of the model each on one GPU
HF_CLASSIFIER_PATH=$1
for i in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$i \
    vllm serve $HF_CLASSIFIER_PATH \
               --task classify \
               --dtype auto \
               --host 0.0.0.0 \
               --port $(( 8000 + $i )) \
               --served-model-name similarity_gpu_$i &
done
