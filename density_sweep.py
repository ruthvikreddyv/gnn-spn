"""
density_sweep.py  –  V2V success probability vs vehicle density
================================================================
Reproduces Figs. 4 and 5 of the paper:
  - V2V success probability vs number of vehicles  (20, 40, 60, 80, 100)
  - Average V2I sum rate vs number of vehicles

Runs both GNN-DDQN baseline and GNN-SPN and plots comparison.

Usage
-----
  python density_sweep.py --n_games 100

Requires both sets of weights to be trained:
  weight/dqn_weights.h5      (from run_baseline.py --mode train)
  weight/GNN_weights.h5      (from run_baseline.py --mode train)
  weight/gnn_spn_best.h5     (from main_spn.py --mode train)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf

from Environment import Environ
from baseline_agent import Agent as BaselineAgent
from agent_spn import AgentSPN, SPNConfig


DENSITIES = [20, 40, 60, 80, 100]


def build_env():
    up    = [3.5/2, 3.5/2+3.5, 250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]
    down  = [250-3.5-3.5/2, 250-3.5/2, 500-3.5-3.5/2, 500-3.5/2, 750-3.5-3.5/2, 750-3.5/2]
    left  = [3.5/2, 3.5/2+3.5, 433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]
    right = [433-3.5-3.5/2, 433-3.5/2, 866-3.5-3.5/2, 866-3.5/2, 1299-3.5-3.5/2, 1299-3.5/2]
    return Environ(down, up, left, right, 750, 1299)


def run_spn_at_density(n_veh, n_games, seed):
    env = build_env()
    env.new_random_game(n_veh)
    cfg = SPNConfig()
    cfg.num_vehicle = n_veh
    agent = AgentSPN(cfg, env)
    v2i, fail = agent.play(n_games=n_games)
    v2v_success = 1.0 - fail
    return v2i, v2v_success


def run_baseline_at_density(n_veh, n_games, seed):
    env = build_env()
    env.new_random_game(n_veh)
    agent = BaselineAgent([], env)
    agent.num_vehicle = n_veh

    V2I_list  = []
    Fail_list = []
    import os
    agent.dqn.model.load_weights('weight/dqn_weights.h5')
    agent.G.G_model.load_weights('weight/GNN_weights.h5')
    agent.training = False
    agent.GraphSAGE = False

    for _ in range(n_games):
        env.new_random_game(n_veh)
        agent.env = env
        agent.neighbor_nodes = []
        agent.action_all_with_power = np.zeros([n_veh, 3, 2], dtype='int32')
        better_state = agent.initial_better_state(0, False)
        Rate_list = []
        for k in range(200):
            import networkx as nx
            for i in range(len(env.vehicles)):
                agent.action_all_with_power[i, :, 0] = -1
                sorted_idx = np.argsort(env.individual_time_limit[i, :])
                for j in sorted_idx:
                    idx = [3*i+j]
                    state_old = agent.get_state([i, j])
                    agent.G.features[3*i+j, :] = state_old[:60]
                    node_emb = agent.G.use_GraphSAGE(
                        agent.channel_reward, 0, idx, False)
                    scale = np.max(np.abs(node_emb)) + 1e-9
                    node_emb = np.squeeze(node_emb / scale)
                    bs = np.concatenate((node_emb, state_old))
                    action = agent.predict(bs, 0, True)
                    agent.merge_action([i, j], action)
                if i % max(1, len(env.vehicles)//5) == 1:
                    reward, percent = env.act_asyn(
                        agent.action_all_with_power.copy())
                    Rate_list.append(np.sum(reward))
        V2I_list.append(np.mean(Rate_list) if Rate_list else 0)
        Fail_list.append(percent)

    return np.mean(V2I_list), 1.0 - np.mean(Fail_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_games",  type=int, default=100)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--spn_only", action="store_true",
                        help="Only run SPN (skip baseline evaluation)")
    args = parser.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    spn_v2i   = []
    spn_v2v   = []
    base_v2i  = []
    base_v2v  = []

    for nv in DENSITIES:
        print(f"\n{'='*50}")
        print(f"  Vehicles: {nv}")
        print(f"{'='*50}")

        print(f"  [GNN-SPN]  evaluating ...")
        v2i_s, v2v_s = run_spn_at_density(nv, args.n_games, args.seed)
        spn_v2i.append(v2i_s)
        spn_v2v.append(v2v_s)
        print(f"  GNN-SPN  V2I={v2i_s:.2f}  V2V_success={v2v_s:.4f}")

        if not args.spn_only:
            print(f"  [Baseline] evaluating ...")
            try:
                v2i_b, v2v_b = run_baseline_at_density(nv, args.n_games, args.seed)
                base_v2i.append(v2i_b)
                base_v2v.append(v2v_b)
                print(f"  Baseline  V2I={v2i_b:.2f}  V2V_success={v2v_b:.4f}")
            except Exception as e:
                print(f"  Baseline failed: {e}")
                base_v2i.append(0)
                base_v2v.append(0)

    # Save results
    np.savez("results/density_sweep.npz",
             densities=DENSITIES,
             spn_v2i=spn_v2i,   spn_v2v=spn_v2v,
             base_v2i=base_v2i, base_v2v=base_v2v)

    # ── Plot V2V success probability (Fig. 4 equivalent) ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(DENSITIES, spn_v2v, "b*-",  linewidth=2, label="GNN-SPN")
    if not args.spn_only and any(v > 0 for v in base_v2v):
        axes[0].plot(DENSITIES, base_v2v, "rs--", linewidth=2, label="GNN-DDQN")
    axes[0].set_xlabel("Number of Vehicles")
    axes[0].set_ylabel("V2V Success Probability")
    axes[0].set_title("V2V Success Probability vs Density")
    axes[0].legend(); axes[0].grid(alpha=0.4)
    axes[0].set_ylim([0.90, 1.0])

    # ── Plot V2I sum rate (Fig. 5 equivalent) ────────────────────────────
    axes[1].plot(DENSITIES, spn_v2i, "b*-",  linewidth=2, label="GNN-SPN")
    if not args.spn_only and any(v > 0 for v in base_v2i):
        axes[1].plot(DENSITIES, base_v2i, "rs--", linewidth=2, label="GNN-DDQN")
    axes[1].set_xlabel("Number of Vehicles")
    axes[1].set_ylabel("Average V2I Sum Rate")
    axes[1].set_title("V2I Sum Rate vs Density")
    axes[1].legend(); axes[1].grid(alpha=0.4)

    plt.suptitle("GNN-SPN vs GNN-DDQN: Density Sweep", fontsize=13)
    plt.tight_layout()
    plt.savefig("results/density_sweep.png", dpi=150)
    print("\nSaved: results/density_sweep.png")
    plt.close()

    # ── Print summary table ───────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"{'Vehicles':>10} {'SPN V2V':>12} {'Base V2V':>12} "
          f"{'SPN V2I':>12} {'Base V2I':>12}")
    print("="*60)
    for k, nv in enumerate(DENSITIES):
        bv = base_v2v[k] if base_v2v else 0
        bi = base_v2i[k] if base_v2i else 0
        print(f"{nv:>10} {spn_v2v[k]:>12.4f} {bv:>12.4f} "
              f"{spn_v2i[k]:>12.2f} {bi:>12.2f}")


if __name__ == "__main__":
    main()
