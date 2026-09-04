"""
agent_spn.py  –  GNN-SPN Best Agent (Two-Phase)
=================================================
Phase 1: Supervised pre-training (2 000 epochs)
  - Build dynamic graph G_t from channel state
  - Generate BER-constrained labels Y*_t  (paper Section III-D)
  - Cross-entropy loss on GAT + MLP policy  (paper Eq. 12)
  - Stores diverse channel realisations in experience replay

Phase 2: PPO fine-tuning (remaining epochs up to 12 000)
  - Reuses the pre-trained GAT trunk (weights frozen for first 500 PPO steps)
  - Adds critic value head
  - PPO-clip update with LOW entropy coef (0.01) — diversity already learned
  - Rollout = one full pass over all 60 V2V links
  - Optimises actual environment reward → closes label-vs-reward gap

Key design decisions
---------------------
1. Soft labels during supervised phase (label_temp=0.5)
   Prevents the policy from becoming overconfident on noisy BER estimates.
   Acts as label smoothing — well-established regularisation trick.

2. GAT trunk frozen for first 500 PPO steps
   Protects the learned interference representations from being destroyed
   by early PPO gradient noise before the critic stabilises.

3. GraphSAGE neighbour embeddings as auxiliary node features
   The baseline's GraphSAGE produces 20-D neighbourhood embeddings that
   capture multi-hop interference context. We concatenate these to the
   raw 82-D state before feeding the GAT. This gives the GAT richer
   starting features without replacing its attention mechanism.
   Result: n_node_features = 82 + 20 = 102.

4. Vehicle-level adjacency weighted by inverse distance
   w_ij = (d_max - d_ij) / d_max — closer vehicles have stronger edges,
   directly encoding interference coupling strength into the graph structure.
"""

from __future__ import print_function, division

import os
import numpy as np
import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt

from Environment import Environ
from Graph_SAGE import GraphSAGE_sup
from spn_model import GNNSPNModel
from label_generator import generate_labels, N_RB, N_POWER, N_ACTIONS
from replay_buffer import SupervisedBuffer, RolloutBuffer


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
class SPNConfig:
    # environment
    num_vehicle: int  = 20
    RB_number:   int  = N_RB       # 20
    n_power:     int  = N_POWER    # 3
    n_actions:   int  = N_ACTIONS  # 60

    # node features: raw state (82) + GraphSAGE embedding (20) = 102
    n_features:  int  = 102

    # GAT encoder
    gat_layers:      int = 2
    gat_hidden_dim:  int = 64
    gat_out_dim:     int = 64
    gat_heads:       int = 4

    # policy MLP head
    mlp_hidden:  int  = 128

    # ── Phase 1: supervised pre-training ──
    sup_epochs:      int   = 2_000    # fast convergence from warm start
    sup_batch:       int   = 64
    replay_capacity: int   = 10_000
    replay_min:      int   = 300
    label_temp:      float = 0.5      # soft label temperature (0 = hard)
    lr_sup:          float = 1e-3
    lr_sup_decay:    float = 0.97
    lr_sup_steps:    int   = 1_000
    lr_minimum:      float = 1e-5

    # ── Phase 2: PPO fine-tuning ──
    ppo_epochs:      int   = 10_000   # total PPO steps after supervised
    ppo_rollout:     int   = 60       # one full V2V pass
    ppo_update_epochs: int = 3
    ppo_minibatch:   int   = 30
    ppo_freeze_steps: int  = 500      # keep GAT trunk frozen initially
    lr_ppo:          float = 3e-4
    entropy_coef:    float = 0.01     # LOW: diversity already learned
    vf_coef:         float = 0.5
    clip_epsilon:    float = 0.2
    gamma:           float = 0.99
    gae_lambda:      float = 0.95

    # misc
    graph_threshold: float = 300.0   # metres for edge construction
    target_sync:     int   = 100
    save_interval:   int   = 500
    log_interval:    int   = 100
    weight_dir:      str   = "weight"


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────
class AgentSPN:

    def __init__(self, config: SPNConfig, environment: Environ):
        self.cfg = config
        self.env = environment
        self.num_vehicle = config.num_vehicle
        self.RB_number   = config.RB_number
        self.V2V_number  = 3 * self.num_vehicle

        os.makedirs(config.weight_dir, exist_ok=True)

        # GraphSAGE for auxiliary embeddings (reuse baseline)
        self.G = GraphSAGE_sup(environment)

        # GNN-SPN model
        self.model = GNNSPNModel(
            n_node_features = config.n_features,
            n_actions       = config.n_actions,
            gat_layers      = config.gat_layers,
            gat_hidden_dim  = config.gat_hidden_dim,
            gat_out_dim     = config.gat_out_dim,
            gat_heads       = config.gat_heads,
            mlp_hidden      = config.mlp_hidden,
            lr_sup          = config.lr_sup,
            lr_sup_decay    = config.lr_sup_decay,
            lr_sup_steps    = config.lr_sup_steps,
            lr_minimum      = config.lr_minimum,
            lr_ppo          = config.lr_ppo,
            entropy_coef    = config.entropy_coef,
            vf_coef         = config.vf_coef,
            clip_epsilon    = config.clip_epsilon,
        )

        # Replay buffers
        self.sup_buf = SupervisedBuffer(
            capacity   = config.replay_capacity,
            n_links    = self.V2V_number,
            n_features = config.n_features,
            n_actions  = config.n_actions,
            n_veh      = config.num_vehicle,
        )
        self.rollout_buf = RolloutBuffer(
            capacity   = config.ppo_rollout,
            n_links    = self.V2V_number,
            n_features = config.n_features,
            n_veh      = config.num_vehicle,
            gamma      = config.gamma,
            gae_lambda = config.gae_lambda,
        )

        # Action bookkeeping
        self.action_all_with_power = np.zeros(
            [config.num_vehicle, 3, 2], dtype="int32")
        self.channel_reward = np.zeros(
            (self.V2V_number, self.RB_number), dtype=np.float32)
        self.neighbor_nodes = []
        self.training = True

        # Logging
        self.sup_losses  = []
        self.ppo_losses  = []
        self.ch_usage    = []

    # ─────────────────────────────────────────────────────────────────────
    # Graph construction helpers
    # ─────────────────────────────────────────────────────────────────────
    def _build_vehicle_adj(self) -> np.ndarray:
        """Weighted adjacency A ∈ R^{n_veh × n_veh}."""
        n = self.num_vehicle
        D = self.env.Distance
        d_max = np.max(D) + 1e-6
        A = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i+1, n):
                if D[i, j] <= self.cfg.graph_threshold:
                    w = (d_max - D[i, j]) / d_max
                    A[i, j] = w; A[j, i] = w
        return A

    def _veh_adj_to_link_adj(self, A_veh: np.ndarray) -> np.ndarray:
        """Expand (n_veh,n_veh) → (k,k) link-level adjacency."""
        n = self.num_vehicle; k = n * 3
        A_link = np.zeros((k, k), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if A_veh[i, j] > 0:
                    for p in range(3):
                        for q in range(3):
                            A_link[3*i+p, 3*j+q] = A_veh[i, j]
        return A_link

    # ─────────────────────────────────────────────────────────────────────
    # Node feature extraction: raw state (82) + GraphSAGE embedding (20)
    # ─────────────────────────────────────────────────────────────────────
    def _build_node_features(self, step=0, use_sage=True) -> np.ndarray:
        """
        Returns X_t ∈ R^{k × 102}.
        Row i = [raw_state(82) || sage_embedding(20)]
        """
        # ── raw state (82-D) ──
        X_raw = np.zeros((self.V2V_number, 82), dtype=np.float32)
        for i in range(len(self.env.vehicles)):
            for j in range(3):
                dest = self.env.vehicles[i].destinations[j]
                V2I  = (self.env.V2I_channels_with_fastfading[i,:] - 80) / 60
                V2V  = (self.env.V2V_channels_with_fastfading[i,dest,:] - 80)/60
                VI   = (-self.env.V2V_Interference_all[i,j,:] - 60) / 60
                NeiS = np.zeros(self.RB_number)
                if self.neighbor_nodes:
                    for nidx in self.neighbor_nodes[3*i+j][0]:
                        rb = self.action_all_with_power[
                            self.G.link[nidx,0],
                            self.G.link[nidx,1]%3, 0]
                        if 0 <= rb < self.RB_number:
                            NeiS[rb] = 1
                dem = np.array([self.env.demand[i,j] / self.env.demand_amount])
                tim = np.array([self.env.individual_time_limit[i,j] /
                                self.env.V2V_limit])
                X_raw[3*i+j] = np.concatenate([V2I, VI, V2V, NeiS, dem, tim])

        # ── GraphSAGE embeddings (20-D) ──
        if use_sage and hasattr(self.G, 'order_nodes') \
                and self.G.order_nodes:
            idx = list(range(self.V2V_number))
            for i in range(len(self.env.vehicles)):
                for j in range(3):
                    self.G.features[3*i+j,:] = X_raw[3*i+j,:60]
            cr = X_raw[:,0:20] + X_raw[:,20:40] - X_raw[:,40:60]
            scale = np.max(np.abs(cr)) + 1e-9
            self.channel_reward = cr / (scale * 1.25)
            emb = self.G.use_GraphSAGE(
                self.channel_reward, step, idx, True)
            emb = emb / (np.max(np.abs(emb)) + 1e-9)
            X = np.concatenate([emb, X_raw], axis=1)   # (k, 102)
        else:
            # Pad with zeros if SAGE not ready yet
            X = np.concatenate(
                [np.zeros((self.V2V_number, 20), dtype=np.float32),
                 X_raw], axis=1)
        return X.astype(np.float32)

    def _init_sage_graph(self, step=0):
        """Initialise GraphSAGE graph structure (same as baseline)."""
        self.G.num_V2V_list = np.zeros(
            (len(self.env.vehicles), len(self.env.vehicles)))
        self.G.link         = np.zeros((self.V2V_number, 2))
        self.G.features     = np.zeros((self.V2V_number, 60))
        self.neighbor_nodes = []
        for i in range(len(self.env.vehicles)):
            for j in range(3):
                self.G.num_V2V_list[
                    i, self.env.vehicles[i].destinations[j]] = 1
        graph, order_nodes, _ = self.G.build_graph(self.G.num_V2V_list)
        self.G.load_graph(graph, order_nodes)
        for i in range(len(self.env.vehicles)):
            for j in range(3):
                node_label = order_nodes[3*i+j]
                neighs = list(nx.neighbors(graph, node_label))
                self.neighbor_nodes.append(
                    [[order_nodes.index(n) for n in neighs]])

    # ─────────────────────────────────────────────────────────────────────
    # Action helpers
    # ─────────────────────────────────────────────────────────────────────
    def _apply_actions(self, actions: np.ndarray):
        for i in range(self.num_vehicle):
            for j in range(3):
                a = int(actions[3*i+j])
                self.action_all_with_power[i,j,0] = a % self.RB_number
                self.action_all_with_power[i,j,1] = a // self.RB_number

    # ─────────────────────────────────────────────────────────────────────
    # Phase 1: Supervised pre-training
    # ─────────────────────────────────────────────────────────────────────
    def _supervised_phase(self):
        print("\n[Phase 1] Supervised pre-training")
        print(f"  epochs      = {self.cfg.sup_epochs}")
        print(f"  label_temp  = {self.cfg.label_temp}  (soft labels)\n")

        self.env.new_random_game(self.num_vehicle)
        self._init_sage_graph(0)
        # Random initial actions for label context
        self.action_all_with_power = np.zeros(
            [self.num_vehicle, 3, 2], dtype="int32")
        self.action_all_with_power[:,:,0] = np.random.randint(
            0, self.RB_number, size=(self.num_vehicle, 3))
        self.action_all_with_power[:,:,1] = np.random.randint(
            0, self.cfg.n_power,  size=(self.num_vehicle, 3))

        epoch_losses = []

        for ep in range(1, self.cfg.sup_epochs + 1):

            if ep % 1000 == 1 and ep > 1:
                self.env.new_random_game(self.num_vehicle)
                self._init_sage_graph(ep)

            # Build inputs
            X_t    = self._build_node_features(step=ep, use_sage=True)
            A_veh  = self._build_vehicle_adj()
            A_link = self._veh_adj_to_link_adj(A_veh)

            # BER-constrained labels
            Y_t = generate_labels(
                v2v_ch_ff      = self.env.V2V_channels_with_fastfading,
                v2v_interf_all = self.env.V2V_Interference_all,
                vehicles       = self.env.vehicles,
                action_all     = self.action_all_with_power,
                n_veh          = self.num_vehicle,
                n_rb           = self.RB_number,
                label_temp     = self.cfg.label_temp,
            )

            # Store in replay
            self.sup_buf.add(X_t, A_veh, Y_t)

            # Supervised update
            loss_val = float('nan')
            if self.sup_buf.ready(self.cfg.replay_min):
                X_b, A_b, Y_b = self.sup_buf.sample(self.cfg.sup_batch)
                bl = []
                for b in range(self.cfg.sup_batch):
                    Al = self._veh_adj_to_link_adj(A_b[b])
                    lv, _ = self.model.supervised_step(
                        tf.constant(X_b[b]),
                        tf.constant(Al),
                        tf.constant(Y_b[b]),
                    )
                    bl.append(float(lv.numpy()))
                loss_val = float(np.mean(bl))
                epoch_losses.append(loss_val)

            # Use model predictions as current actions
            if ep > self.cfg.replay_min // 2:
                actions, _, _ = self.model.get_action(X_t, A_link)
                self._apply_actions(actions)
                self.env.Compute_Interference(self.action_all_with_power)

            # Advance environment
            self.env.renew_positions()
            self.env.renew_channels_fastfading()
            self.G.update_target_network() \
                if ep % self.cfg.target_sync == 0 else None

            if ep % self.cfg.log_interval == 0:
                n_used = len(np.unique(
                    self.action_all_with_power[:,:,0]))
                mean_l = np.nanmean(
                    epoch_losses[-self.cfg.log_interval:]) \
                    if epoch_losses else float('nan')
                print(f"  [sup ep {ep:>5}]  loss={mean_l:.4f}  "
                      f"channels={n_used}/20")
                self.sup_losses.append(mean_l)

            if ep % self.cfg.save_interval == 0:
                self.model.save_weights_to_file(
                    os.path.join(self.cfg.weight_dir, "gnn_spn_best.h5"))

        self.model.save_weights_to_file(
            os.path.join(self.cfg.weight_dir, "gnn_spn_sup.h5"))
        print("\n[Phase 1] Complete.")

    # ─────────────────────────────────────────────────────────────────────
    # Phase 2: PPO fine-tuning
    # ─────────────────────────────────────────────────────────────────────
    def _ppo_phase(self):
        print("\n[Phase 2] PPO fine-tuning")
        print(f"  steps        = {self.cfg.ppo_epochs}")
        print(f"  freeze_steps = {self.cfg.ppo_freeze_steps}  (GAT trunk)")
        print(f"  entropy_coef = {self.cfg.entropy_coef}\n")

        self.env.new_random_game(self.num_vehicle)
        self._init_sage_graph(0)

        ppo_step = 0
        for step in range(1, self.cfg.ppo_epochs + 1):

            if step % 2000 == 1 and step > 1:
                self.env.new_random_game(self.num_vehicle)
                self._init_sage_graph(step)

            # Freeze GAT trunk for first ppo_freeze_steps
            freeze = (step <= self.cfg.ppo_freeze_steps)
            if freeze:
                for layer in self.model.gat.layers:
                    layer.trainable = False
            else:
                for layer in self.model.gat.layers:
                    layer.trainable = True

            # ── collect one full rollout (60 links) ──
            self.rollout_buf.clear()
            reward_sum = 0.0

            X_t    = self._build_node_features(step=step)
            A_veh  = self._build_vehicle_adj()
            A_link = self._veh_adj_to_link_adj(A_veh)

            actions, log_probs, value = self.model.get_action(X_t, A_link)
            self._apply_actions(actions)

            reward_all = 0.0
            for i in range(self.num_vehicle):
                for j in range(3):
                    r = self.env.act_for_training(
                        self.action_all_with_power, [i,j])
                    reward_all += r
                    self.channel_reward[3*i+j,
                        self.action_all_with_power[i,j,0]] = r

            reward_mean = reward_all / self.V2V_number
            done = False

            self.rollout_buf.add(
                X_t, A_veh, actions, log_probs,
                reward_mean, value, done)

            self.env.renew_positions()
            self.env.renew_channels_fastfading()
            self.env.Compute_Interference(self.action_all_with_power)

            # ── PPO update ──
            if self.rollout_buf.ready():
                ppo_step += 1
                # bootstrap value
                X_new    = self._build_node_features(step=step)
                A_new    = self._build_vehicle_adj()
                Al_new   = self._veh_adj_to_link_adj(A_new)
                _, _, last_v = self.model.get_action(X_new, Al_new)

                (Xb, Ab, act_b, lp_b,
                 ret_b, adv_b) = self.rollout_buf.compute_gae(last_v)

                losses = []
                for _ in range(self.cfg.ppo_update_epochs):
                    for b_idx in range(len(Xb)):
                        Al_b = self._veh_adj_to_link_adj(Ab[b_idx])
                        ld = self.model.ppo_step(
                            tf.constant(Xb[b_idx]),
                            tf.constant(Al_b),
                            tf.constant(act_b[b_idx]),
                            tf.constant(lp_b[b_idx]),
                            tf.constant(ret_b[b_idx:b_idx+1]),
                            tf.constant(adv_b[b_idx:b_idx+1]),
                        )
                        losses.append(float(ld["total"].numpy()))

                n_used = len(np.unique(
                    self.action_all_with_power[:,:,0]))
                self.ppo_losses.append(float(np.mean(losses)))
                self.ch_usage.append(n_used)

                if step % self.cfg.log_interval == 0:
                    print(f"  [ppo step {step:>5}]  "
                          f"loss={np.mean(losses):.4f}  "
                          f"channels={n_used}/20  "
                          f"{'[frozen]' if freeze else ''}")

            # GNN target sync
            if step % self.cfg.target_sync == 0:
                self.G.update_target_network()
                self.G.G_model.save_weights(
                    os.path.join(self.cfg.weight_dir, "GNN_weights.h5"))

            if step % self.cfg.save_interval == 0:
                self.model.save_weights_to_file(
                    os.path.join(self.cfg.weight_dir, "gnn_spn_best.h5"))

        self.model.save_weights_to_file(
            os.path.join(self.cfg.weight_dir, "gnn_spn_best.h5"))
        print("\n[Phase 2] Complete.")

    # ─────────────────────────────────────────────────────────────────────
    # Main training entry point
    # ─────────────────────────────────────────────────────────────────────
    def train(self):
        self._supervised_phase()
        self._ppo_phase()
        self._plot_training_curves()
        print("\n[GNN-SPN] Full training complete.")

    # ─────────────────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────────────────
    def play(self, n_games=100):
        print("\n[GNN-SPN] === Evaluation ===")
        self.model.load_weights_from_file(
            os.path.join(self.cfg.weight_dir, "gnn_spn_best.h5"))

        V2I_Rate_list     = np.zeros(n_games)
        Fail_percent_list = np.zeros(n_games)
        power_sel = [[], [], []]

        for game_idx in range(n_games):
            self.env.new_random_game(self.num_vehicle)
            self._init_sage_graph(0)
            self.action_all_with_power = np.zeros(
                [self.num_vehicle, 3, 2], dtype="int32")
            Rate_list = []
            print(f"  game {game_idx}")

            for t in range(200):
                X_t    = self._build_node_features(step=t, use_sage=False)
                A_veh  = self._build_vehicle_adj()
                A_link = self._veh_adj_to_link_adj(A_veh)
                actions, _, _ = self.model.get_action(
                    X_t, A_link, deterministic=True)
                self._apply_actions(actions)

                for i in range(self.num_vehicle):
                    for j in range(3):
                        trem = X_t[3*i+j, 101]
                        if trem > 0:
                            p = int(self.action_all_with_power[i,j,1])
                            power_sel[p].append(float(trem))

                if t % max(1, self.num_vehicle // 5) == 1:
                    reward, percent = self.env.act_asyn(
                        self.action_all_with_power.copy())
                    Rate_list.append(float(np.sum(reward)))

                self.env.renew_positions()
                self.env.renew_channels_fastfading()
                self.env.Compute_Interference(self.action_all_with_power)

            V2I_Rate_list[game_idx]     = np.mean(Rate_list) if Rate_list else 0
            Fail_percent_list[game_idx] = percent

        self._plot_power_prob(power_sel)
        self._plot_vehicle_dist(power_sel)

        mean_v2i  = float(np.mean(V2I_Rate_list))
        mean_fail = float(np.mean(Fail_percent_list))
        print(f"\n[GNN-SPN] Mean V2I rate : {mean_v2i:.4f}")
        print(f"[GNN-SPN] Mean fail %   : {mean_fail:.4f}")
        return mean_v2i, mean_fail

    # ─────────────────────────────────────────────────────────────────────
    # Plots
    # ─────────────────────────────────────────────────────────────────────
    def _plot_training_curves(self):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        if self.sup_losses:
            axes[0].plot(self.sup_losses, color="steelblue")
            axes[0].set_title("Supervised loss")
            axes[0].set_xlabel("Epoch (×100)")
            axes[0].set_ylabel("Cross-entropy")
            axes[0].grid(alpha=0.4)

        if self.ppo_losses:
            axes[1].plot(self.ppo_losses, color="darkorange")
            axes[1].set_title("PPO loss")
            axes[1].set_xlabel("PPO step")
            axes[1].grid(alpha=0.4)

        if self.ch_usage:
            axes[2].plot(self.ch_usage, color="green")
            axes[2].axhline(20, ls="--", color="gray", alpha=0.5)
            axes[2].set_title("Channel diversity")
            axes[2].set_xlabel("PPO step")
            axes[2].set_ylabel("Channels used / 20")
            axes[2].set_ylim([0, 21])
            axes[2].grid(alpha=0.4)

        plt.suptitle("GNN-SPN Training Curves", fontsize=13)
        plt.tight_layout()
        plt.savefig("training_loss_curve.png", dpi=150)
        plt.close()
        print("[GNN-SPN] Saved: training_loss_curve.png")

    def _plot_power_prob(self, power_sel):
        bin_edges = None; bars = []
        for pl in power_sel:
            n, be = np.histogram(pl if pl else [0], bins=10, range=(0, 0.11))
            bars.append(n)
            if bin_edges is None: bin_edges = be
        total = bars[0]+bars[1]+bars[2]+1e-9
        plt.figure()
        plt.plot(bin_edges[:-1]*0.1+0.01, bars[0]/total, "b*-", label="23 dB")
        plt.plot(bin_edges[:-1]*0.1+0.01, bars[1]/total, "rs-", label="10 dB")
        plt.plot(bin_edges[:-1]*0.1+0.01, bars[2]/total, "go-", label="5 dB")
        plt.xlim([0,0.12]); plt.legend(); plt.grid()
        plt.xlabel("Time left (s)"); plt.ylabel("Power selection probability")
        plt.title("GNN-SPN Power Selection"); plt.savefig("GNN-SPN.png", dpi=150)
        plt.close()

    def _plot_vehicle_dist(self, power_sel):
        combined = sum(power_sel, [])
        y, be = np.histogram(combined if combined else [0], bins=10, range=(0,0.11))
        bw = (be[1]-be[0])*0.1
        plt.figure()
        plt.bar(be[:-1]*0.1+0.01, y, width=bw, color="c")
        plt.xlim([0,0.12]); plt.grid()
        plt.xlabel("Time left (s)"); plt.ylabel("Vehicle Distribution")
        plt.title("GNN-SPN Vehicle Distribution")
        plt.savefig("GNN-SPN_y0.png", dpi=150); plt.close()
