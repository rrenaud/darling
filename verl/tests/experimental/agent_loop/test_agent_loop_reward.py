import os

import ray
from hydra import compose, initialize_config_dir
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from tests.experimental.agent_loop.agent_utils import init_agent_loop_manager
from verl.protocol import DataProto
from verl.trainer.main_ppo import create_rl_sampler
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn


def test_agent_loop_compute_score():
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        }
    )

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose("ppo_trainer")

    model_path = "Qwen/Qwen2.5-1.5B-Instruct"
    config.data.return_raw_chat = True
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.actor.use_dynamic_bsz = True
    config.actor_rollout_ref.rollout.name = os.environ["ROLLOUT_NAME"]
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.prompt_length = 1024
    config.actor_rollout_ref.rollout.response_length = 4096

    # 1. init agent loop manager
    agent_loop_manager = init_agent_loop_manager(config)

    # 2. init dataset and dataloader
    local_folder = os.path.expanduser("~/verl-data/gsm8k/")
    data_files = [os.path.join(local_folder, "train.parquet")]
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    dataset = RLHFDataset(
        data_files=data_files,
        tokenizer=tokenizer,
        config=config.data,
        processor=None,
    )

    batch_size = 128
    sampler = create_rl_sampler(config.data, dataset)
    dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=config.data.dataloader_num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        sampler=sampler,
    )

    # 3. generate_sequences with agent loop
    batch_dict = next(iter(dataloader))
    batch = DataProto.from_single_dict(batch_dict)
    gen_batch = agent_loop_manager.generate_sequences(prompts=batch)

    rm_scores = gen_batch.batch["rm_scores"]
    sample_scores = rm_scores.sum(dim=1)
    assert sample_scores.min() == 0.0, f"min score: {sample_scores.min()}"
    assert sample_scores.max() == 1.0, f"max score: {sample_scores.max()}"
    print(f"gsm8k acc: {sample_scores.mean()}")

    print("Test passed!")
    ray.shutdown()
