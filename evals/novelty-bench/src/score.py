from numpyencoder import NumpyEncoder

import argparse
import asyncio
import bisect
import functools
import json
import os

import datasets
import numpy as np
import torch
from aiofiles import open as aio_open
from datasets import load_dataset
from pydantic import BaseModel
from tqdm.asyncio import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, LlamaPreTrainedModel, LlamaModel
CONCURRENT_REQUESTS = 1

def transform_raw_reward_athene(score):
    """
    Map a single raw score to an integer between 1 and 10,
    using the precomputed decile cutoffs:
      10th percentile: -4.6875
      20th percentile: -4.1562
      30th percentile: -3.4844
      40th percentile: -2.7812
      50th percentile: -2.1719
      60th percentile: -1.5781
      70th percentile: -0.9141
      80th percentile: -0.2295
      90th percentile:  0.4414
    """
    if score <= -4.6875:
        return 1
    elif score <= -4.1562:
        return 2
    elif score <= -3.4844:
        return 3
    elif score <= -2.7812:
        return 4
    elif score <= -2.1719:
        return 5
    elif score <= -1.5781:
        return 6
    elif score <= -0.9141:
        return 7
    elif score <= -0.2295:
        return 8
    elif score <=  0.4414:
        return 9
    else:
        return 10

reward_thresholds = [
    -7.71875,
    -6.28125,
    -6.0,
    -5.71875,
    -5.5,
    -5.0,
    -4.375,
    -3.4375,
    -2.046875,
]

class AtheneForSequenceClassification(LlamaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.v_head = torch.nn.Linear(config.hidden_size, 1, bias=False)
        self.CLS_ID = 128003  # The ID of the CLS token in the Athene model
        self.post_init()

    def forward(self, input_ids=None, past_key_values=None, attention_mask=None, position_ids=None):
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]  # Get the last hidden state
        rewards = self.v_head(hidden_states).squeeze(-1)  # Apply the value head

        bs = int(input_ids.shape[0])
        scores = []
        for i in range(bs):
            c_inds = (input_ids[i] == self.CLS_ID).nonzero()
            c_ind = c_inds[-1].item()
            scores.append(rewards[i, c_ind])
        scores = torch.stack(scores)
        return {"scores": scores}


def transform_raw_reward(reward: float) -> int:
    # score of 1 to 10
    return bisect.bisect_left(reward_thresholds, reward) + 1


@functools.cache
def rm_and_tokenizer():
    # Load model and tokenizer
    model_name = "Skywork/Skywork-Reward-Gemma-2-27B-v0.2"
    #model_name = "Nexusflow/Athene-RM-70B"
    #model_name = "/checkpoint/ram/tianjian/reward_models/Athene-RM-8B"
    if "Athene" not in model_name:
        rm = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            num_labels=1,
        )
    else:
        rm = AtheneForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return rm, tokenizer


class Rating(BaseModel):
    rating: int


@torch.inference_mode()
async def score_partition_rm(
    prompt: str, generations: list[str], partition: list[int]
) -> tuple[list[int], list[int]]:
    """Asynchronously scores the partition."""
    rm, tokenizer = rm_and_tokenizer()
    convs = [
        [
            {"content": prompt, "role": "user"},
            {"content": generation, "role": "assistant"},
        ]
        for generation in generations
    ]

    formatted = tokenizer.apply_chat_template(convs, tokenize=False)

    # append tokenizer.cls_token to every input
    #formatted = [generation + tokenizer.cls_token for generation in formatted]

    batch = tokenizer(
        formatted,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=4096,
    )

    # Get the reward scores
    with torch.no_grad():
        raw_rewards = rm(**batch).logits[:, 0].tolist()
        #raw_scores = rm(**batch) 
    #raw_rewards = raw_scores["scores"].cpu().tolist()
    scores = [transform_raw_reward(r) for r in raw_rewards]
    #print(f"Scores: {scores}")
    
    generation_scores = []
    partition_scores = []

    for s, p in zip(scores, partition, strict=False):
        if p == len(partition_scores):
            generation_scores.append(s)
            partition_scores.append(s)
        else:
            #generation_scores.append(0)
            generation_scores.append(s)

    assert len(partition_scores) == (max(partition) + 1), (
        f"partition_scores: {partition_scores}, partition: {partition}"
    )
    return generation_scores, partition_scores


async def process_instances(instances, output_file, patience):
    """Processes all instances concurrently and writes results to a file."""
    # Check if file exists and has matching keys
    if os.path.exists(output_file):
        try:
            existing_output = load_dataset("json", data_files=output_file, split="train")
            if not set(instances["id"]) - set(existing_output["id"]):
                print("All prompts are scored. Skipping.")
                return
        except datasets.exceptions.DatasetGenerationError:
            pass

    async with aio_open(output_file, "w", buffering=1) as f:
        semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

        async def process_single_instance(instance):
            async with semaphore:
                generation_scores, partition_scores = await score_partition_rm(
                    instance["prompt"],
                    instance["generations"],
                    instance["partition"],
                )
                avg_quality = np.average(
                    generation_scores,
                    #weights=patience ** np.arange(len(instance["generations"])),
                )

                utility = np.sum(partition_scores) / len(instance["generations"]) # no patience weighting here

                return {
                    **instance,
                    "generation_scores": generation_scores,
                    "partition_scores": partition_scores,
                    "quality": avg_quality,
                    "utility": utility,
                }

        tasks = [process_single_instance(instance) for instance in instances]
        
        for result in tqdm(await asyncio.gather(*tasks), total=len(instances)):
            await f.write(json.dumps(result, cls=NumpyEncoder) + "\n")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-dir", help="Directory containing evaluation files", required=True
    )
    parser.add_argument(
        "--patience",
        help="Discount factor for computing cumulative utility.",
        type=float,
        default=0.8,
    )

    args = parser.parse_args()

    eval_dir = args.eval_dir
    instances = load_dataset(
        "json",
        data_files=os.path.join(eval_dir, "partitions.jsonl"),
        split="train",
    )

    os.makedirs(eval_dir, exist_ok=True)

    output_file = os.path.join(eval_dir, "scores.jsonl")
    await process_instances(instances, output_file, args.patience)


if __name__ == "__main__":
    asyncio.run(main())
