import argparse
import json
import os
import warnings
from typing import Any

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PretrainedConfig


def _init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model_path", type=str, required=True, help="The path for the huggingface model")
    parser.add_argument("--new_config_path", type=str, required=True, help="The path for the new config file")
    parser.add_argument("--output_path", type=str, required=True, help="The path for the output random model")
    args = parser.parse_args()
    return args


def check_output_path(output_path: str):
    if os.path.exists(output_path):
        warnings.warn(f"Output path '{output_path}' already exists. Will do nothing.", stacklevel=2)
        exit()
    else:
        os.makedirs(output_path, exist_ok=True)
        print(f"Output path '{output_path}' created.")


def check_configs(original_config: dict[str, Any], new_config: dict[str, Any]) -> bool:
    """
    Check if the original config and new config are compatible.
    This is a placeholder function; actual implementation may vary based on requirements.
    """
    # Example check: ensure 'model_type' is the same
    if new_config.get("model_type", None) is not None and original_config.get("model_type") != new_config.get(
        "model_type"
    ):
        raise RuntimeError("Model types do not match.")
    for key in new_config:
        if key not in original_config:
            warnings.warn(
                f"Key '{key}' in new config does not exist in original config, may not take effect.", stacklevel=2
            )


def init_random_model(hf_model_path, new_config_path, output_path):
    config = AutoConfig.from_pretrained(hf_model_path)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    config_dict = PretrainedConfig.get_config_dict(hf_model_path)[0]
    print(config_dict)
    with open(new_config_path) as f:
        new_config_dict = json.load(f)
    check_configs(config_dict, new_config_dict)
    config_dict.update(new_config_dict)
    new_confg = config.from_dict(config_dict)
    print(f"new_config: {new_confg}")
    model = AutoModelForCausalLM.from_config(new_confg)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    new_confg.save_pretrained(output_path)
    print(f"Random model initialized and saved to {output_path}")


if __name__ == "__main__":
    args = _init_args()
    check_output_path(args.output_path)
    init_random_model(
        hf_model_path=args.hf_model_path, new_config_path=args.new_config_path, output_path=args.output_path
    )
