source ~/.bashrc
source ~/.zshrc
conda init
conda activate /checkpoint/ram/tianjian/verl_env

# Set NCCL environment variables
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
mapfile -t nodes_array < <(scontrol show hostnames "$SLURM_JOB_NODELIST")

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == " " ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6332
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

printenv

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
        ray start --head --node-ip-address="$head_node_ip" --port=$port \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
            ray start --address "$ip_head" --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
    sleep 5
done

B=32
N=8
L=1024

PYTHONBUFFERED=1 srun --overlap --nodes 1 --ntasks=1 -w "$head_node" \
    python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/checkpoint/ram/tianjian/wildchat10k.parquet \
    data.val_files=/checkpoint/ram/tianjian/creative_writing_data/valid.parquet \
    data.prompt_key="prompt" \
    data.train_batch_size=${B} \
    data.max_prompt_length=512 \
    data.max_response_length=${L} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/datasets/pretrained-llms/Llama-3.1-8B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40000 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=${N} \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=/home/$USER/diverse_responses/verl/verl/utils/reward_score/diversity_rewards.py \
    custom_reward_function.name=ngram \
    reward_model.reward_manager=diversity \
    reward_model.enable=True \
    reward_model.model.input_tokenizer=/checkpoint/ram/tianjian/reward_models/Athene-RM-8B \
    reward_model.model.path=/checkpoint/ram/tianjian/reward_models/Athene-RM-8B \
    reward_model.micro_batch_size_per_gpu=16 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='athene' \
    trainer.experiment_name=quality_ngram_multiplicative_b${B}_n${N}_l${L} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.save_freq=200 \
    trainer.test_freq=200 \
    trainer.default_local_dir=/checkpoint/ram/tianjian/checkpoints_skywork/wildchat10k_quality_ngram_multiplicative_b${B}_n${N}_l${L} \
    trainer.validation_data_dir=/checkpoint/ram/tianjian/checkpoints_skywork/wildchat10k_quality_ngram_multiplicative_b${B}_n${N}_l${L}/rollouts \
    trainer.total_epochs=10 $@ +reward_model.multiplicative=True
