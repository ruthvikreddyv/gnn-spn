# GNN-SPN: Full Project

Graph Neural Network with Supervised Policy Network for V2X Resource Allocation.
Extends the GNN-DDQN baseline (IEEE 10697115) with a two-phase training approach:
**Supervised Pre-training → PPO Fine-tuning**.

---

## Folder Structure

```
GNN_SPN_FULL/
│
├── ── NEW: GNN-SPN implementation ──
│   ├── gat_encoder.py        Multi-head GAT encoder (paper Section III-B)
│   ├── label_generator.py    BER-constrained label generation (Section III-D)
│   ├── spn_model.py          GNN-SPN model: GAT + policy head + value head
│   ├── replay_buffer.py      Supervised buffer + PPO rollout buffer with GAE
│   ├── agent_spn.py          Two-phase agent: supervised → PPO fine-tuning
│   └── main_spn.py           Entry point (train / play / compare)
│
├── ── BASELINE: GNN-DDQN (unchanged) ──
│   ├── baseline_agent.py     Original agent.py (renamed)
│   ├── baseline_main.py      Original main.py  (renamed)
│   ├── Environment.py        V2X simulation environment
│   ├── Graph_SAGE.py         GraphSAGE implementation
│   ├── model_Graph.py        Graph model layers
│   ├── base.py               Base model utilities
│   ├── dqn_model.py          DDQN model
│   └── replay_memory.py      DDQN replay memory
│
├── ── UTILITIES ──
│   ├── run_baseline.py       Train/evaluate GNN-DDQN baseline
│   ├── density_sweep.py      Reproduce Figs. 4 & 5 (20–100 vehicles)
│   └── requirements.txt      Python dependencies
│
├── weight/                   Saved model weights (created at runtime)
├── images/                   Place paper figures here for LaTeX
└── results/                  Evaluation outputs saved here
```

---

## Setup

```bash
# Python 3.8 required
pip install -r requirements.txt

# GPU version of TensorFlow 2.6 (recommended)
# CUDA 11.2 + cuDNN 8.1 for TF 2.6 GPU
```

---

## Step-by-Step: Reproduce Paper Results

### Step 1 — Train the baseline (GNN-DDQN)

```bash
python run_baseline.py --mode train
```
This saves `weight/dqn_weights.h5` and `weight/GNN_weights.h5`.

### Step 2 — Evaluate baseline and save results

```bash
python run_baseline.py --mode play
```
Note the printed Mean V2I rate and Mean fail percent, then save:
```python
import numpy as np
np.savez('results/baseline_results.npz', v2i=MEAN_V2I, fail=MEAN_FAIL)
```

### Step 3 — Train GNN-SPN (two-phase)

```bash
python main_spn.py --mode train
```

Training has two phases printed separately:
```
[Phase 1] Supervised pre-training
  [sup ep   100]  loss=0.8432  channels=19/20
  [sup ep   200]  loss=0.3241  channels=20/20
  ...
  [sup ep  2000]  loss=0.1123  channels=20/20

[Phase 2] PPO fine-tuning
  [ppo step   100]  loss=0.2341  channels=20/20
  ...
```

Saves `weight/gnn_spn_best.h5` every 500 steps.

### Step 4 — Evaluate GNN-SPN

```bash
python main_spn.py --mode play --n_games 100
```

Produces:
- `GNN-SPN.png`         — power selection probability plot
- `GNN-SPN_y0.png`      — vehicle distribution plot
- `spn_results.npz`     — numerical results

### Step 5 — Compare and generate paper figures

```bash
python main_spn.py --mode compare
```
Produces `comparison_DDQN_vs_SPN.png`.

### Step 6 — Density sweep (Figs. 4 & 5)

```bash
python density_sweep.py --n_games 100
```
Evaluates both models at 20, 40, 60, 80, 100 vehicles.
Produces `results/density_sweep.png` and prints the summary table.

---

## Configuration (agent_spn.py → SPNConfig)

| Parameter | Default | Description |
|---|---|---|
| `sup_epochs` | 2000 | Supervised pre-training epochs |
| `ppo_epochs` | 10000 | PPO fine-tuning steps |
| `label_temp` | 0.5 | Soft label temperature (0=hard one-hot) |
| `entropy_coef` | 0.01 | PPO entropy bonus (low — diversity from supervised) |
| `ppo_freeze_steps` | 500 | Steps to freeze GAT trunk at PPO start |
| `gat_layers` | 2 | Number of GAT encoder layers |
| `gat_heads` | 4 | Attention heads per GAT layer |
| `gat_out_dim` | 64 | GAT output embedding dimension |
| `mlp_hidden` | 128 | Policy MLP hidden size |
| `lr_sup` | 1e-3 | Supervised phase learning rate |
| `lr_ppo` | 3e-4 | PPO learning rate |
| `gamma` | 0.99 | Reward discount factor |
| `gae_lambda` | 0.95 | GAE-λ for advantage estimation |

To scale to more vehicles:
```bash
python main_spn.py --mode train --num_vehicle 40
```

---

## Architecture Summary

```
Channel state (Environment.py)
        │
        ├─→ GraphSAGE embeddings (20-D)    [Graph_SAGE.py]
        └─→ Raw state features  (82-D)
                │
                └─→ Concatenate → X_t (102-D per link)
                          │
                    GAT Encoder             [gat_encoder.py]
                    (2 layers, 4 heads)
                    + Residual + LayerNorm
                          │
                        Z_t (64-D per link)
                          │
              ┌───────────┴───────────┐
          Policy head             Value head
         (MLP 128-D)             (MLP 128-D)
              │                      │
          π(a|s) ∈ R^60          V(s) scalar
              │
        ┌─────┴──────┐
   Phase 1:      Phase 2:
   Cross-entropy  PPO-clip
   on BER labels  on env reward
```

---

## Expected Results (20 vehicles)

| Metric | GNN-DDQN | GNN-SPN |
|---|---|---|
| V2V success prob. | 0.978 | 0.982 |
| V2I rate (Mbps)  | 178   | 185   |
| Channels used    | ~19   | 20    |

---

## Paper Reference

> "Enabling 6G Ultra-Reliable V2X Through Constraint-Preserving Graph
> Learning and Scalable Resource Allocation"
> IEEE 2024. DOI: 10.1109/10697115

Baseline:
> GNN-DDQN: https://ieeexplore.ieee.org/document/10697115
