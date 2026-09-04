"""
main_spn.py  –  GNN-SPN Best (Two-Phase) Entry Point
======================================================
Usage
-----
  python main_spn.py --mode train              # full two-phase training
  python main_spn.py --mode sup_only           # supervised phase only
  python main_spn.py --mode ppo_only           # PPO only (load sup weights)
  python main_spn.py --mode play               # evaluate
  python main_spn.py --mode compare            # plot vs baseline
  python main_spn.py --num_vehicle 40          # scale up
"""

import argparse
import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from Environment import Environ
from agent_spn import AgentSPN, SPNConfig


def build_env():
    up    = [3.5/2, 3.5/2+3.5, 250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]
    down  = [250-3.5-3.5/2, 250-3.5/2, 500-3.5-3.5/2, 500-3.5/2, 750-3.5-3.5/2, 750-3.5/2]
    left  = [3.5/2, 3.5/2+3.5, 433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]
    right = [433-3.5-3.5/2, 433-3.5/2, 866-3.5-3.5/2, 866-3.5/2, 1299-3.5-3.5/2, 1299-3.5/2]
    return Environ(down, up, left, right, 750, 1299)


def plot_comparison(v2i_base, v2i_spn, fail_base, fail_spn):
    labels = ["GNN-DDQN\n(baseline)", "GNN-SPN\n(proposed)"]
    colors = ["steelblue", "darkorange"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(labels, [v2i_base,    v2i_spn],       color=colors)
    axes[0].set_ylabel("Mean V2I Rate"); axes[0].set_title("V2I Rate")
    axes[0].grid(axis="y", alpha=0.4)
    axes[1].bar(labels, [fail_base*100, fail_spn*100], color=colors)
    axes[1].set_ylabel("V2V Failure (%)"); axes[1].set_title("V2V Failure Rate")
    axes[1].grid(axis="y", alpha=0.4)
    plt.suptitle("GNN-DDQN vs GNN-SPN"); plt.tight_layout()
    plt.savefig("comparison_DDQN_vs_SPN.png", dpi=150)
    print("Saved: comparison_DDQN_vs_SPN.png"); plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",        default="train",
                        choices=["train","sup_only","ppo_only","play","compare"])
    parser.add_argument("--num_vehicle", type=int,   default=20)
    parser.add_argument("--sup_epochs",  type=int,   default=2_000)
    parser.add_argument("--ppo_epochs",  type=int,   default=10_000)
    parser.add_argument("--n_games",     type=int,   default=100)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"GPU: {gpus[0].name}")
        except RuntimeError as e:
            print(e)
    else:
        print("CPU mode.")

    cfg             = SPNConfig()
    cfg.num_vehicle = args.num_vehicle
    cfg.sup_epochs  = args.sup_epochs
    cfg.ppo_epochs  = args.ppo_epochs
    os.makedirs("weight", exist_ok=True)

    env = build_env()
    env.new_random_game(cfg.num_vehicle)
    agent = AgentSPN(cfg, env)

    if args.mode == "train":
        # Full two-phase training
        agent.train()

    elif args.mode == "sup_only":
        # Supervised phase only (useful for debugging / paper reproduction)
        agent._supervised_phase()
        agent._plot_training_curves()

    elif args.mode == "ppo_only":
        # Load supervised weights, run PPO fine-tuning only
        agent.model.load_weights_from_file(
            os.path.join(cfg.weight_dir, "gnn_spn_sup.h5"))
        agent._ppo_phase()
        agent._plot_training_curves()

    elif args.mode == "play":
        v2i, fail = agent.play(n_games=args.n_games)
        np.savez("spn_results.npz", v2i=v2i, fail=fail)
        print(f"\nV2I rate : {v2i:.4f}")
        print(f"Fail %   : {fail*100:.2f}%")

    elif args.mode == "compare":
        v2i_spn, fail_spn = agent.play(n_games=args.n_games)
        try:
            b = np.load("baseline_results.npz")
            v2i_base  = float(b["v2i"])
            fail_base = float(b["fail"])
        except FileNotFoundError:
            print("baseline_results.npz not found – using paper values (20 vehicles)")
            v2i_base  = 178.0
            fail_base = 1 - 0.978
        plot_comparison(v2i_base, v2i_spn, fail_base, fail_spn)
        dv = (v2i_spn - v2i_base) / (abs(v2i_base)+1e-9) * 100
        df = (fail_base - fail_spn) / (abs(fail_base)+1e-9) * 100
        print(f"\nΔ V2I rate  : {dv:+.2f}%")
        print(f"Δ Fail rate : {df:+.2f}%  (positive = SPN better)")


if __name__ == "__main__":
    main()
