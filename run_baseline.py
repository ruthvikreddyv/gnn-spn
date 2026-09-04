"""
run_baseline.py  –  Run GNN-DDQN baseline and save results
============================================================
Run this BEFORE training GNN-SPN to get baseline numbers for comparison.

Usage
-----
  # Train baseline
  python run_baseline.py --mode train

  # Evaluate baseline and save results for comparison
  python run_baseline.py --mode play
"""

import argparse
import random
import numpy as np
import tensorflow as tf

from Environment import Environ
from baseline_agent import Agent


def build_env():
    up    = [3.5/2, 3.5/2+3.5, 250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]
    down  = [250-3.5-3.5/2, 250-3.5/2, 500-3.5-3.5/2, 500-3.5/2, 750-3.5-3.5/2, 750-3.5/2]
    left  = [3.5/2, 3.5/2+3.5, 433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]
    right = [433-3.5-3.5/2, 433-3.5/2, 866-3.5-3.5/2, 866-3.5/2, 1299-3.5-3.5/2, 1299-3.5/2]
    return Environ(down, up, left, right, 750, 1299)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="play", choices=["train", "play"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    env = build_env()
    env.new_random_game(20)
    agent = Agent([], env)

    if args.mode == "train":
        print("[Baseline] Training GNN-DDQN ...")
        agent.train()

    elif args.mode == "play":
        print("[Baseline] Evaluating GNN-DDQN ...")
        # agent.play() returns mean V2I rate and fail percent
        # It also saves GNN-DDQN.png and GNN-DDQN_y0.png
        agent.play()

        # Read back logged values from agent
        # (agent.play() prints them — capture manually or parse output)
        print("\n[Baseline] Save results to baseline_results.npz manually:")
        print("  import numpy as np")
        print("  np.savez('results/baseline_results.npz',")
        print("           v2i=MEAN_V2I_VALUE, fail=MEAN_FAIL_VALUE)")


if __name__ == "__main__":
    main()
