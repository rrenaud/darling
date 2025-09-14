import argparse

import numpy as np


def extract_reward_from_line(line):
    # TODO: this function needs error handling
    try:
        key_vals = line.split(" - ")
        for key_val in key_vals:
            key, val = key_val.split(":")
            if key == "critic/rewards/mean":
                reward = float(val)
                return reward
        return -np.inf
    except Exception:
        return -np.inf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", required=True, type=str)
    parser.add_argument("--target", type=float, default=0.2, help="target reward score")

    args = parser.parse_args()

    with open(args.output_file) as f:
        output = f.read().split("\n")

    best_reward = -np.inf
    for line in output:
        if line.startswith("step"):
            reward = extract_reward_from_line(line)
            if reward > best_reward:
                best_reward = reward

    print(f"Best reward is {best_reward}")
    assert best_reward > args.target, f"Best reward must be greater than {args.target}. best_reward: {best_reward}"
    print("Check passes")
