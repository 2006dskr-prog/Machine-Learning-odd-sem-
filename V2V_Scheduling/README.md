# Multi-Agent PPO for Semantic and AoI-Aware V2V Scheduling

A research prototype that trains six vehicle agents with MAPPO to
schedule message transmissions over a shared V2V channel, jointly
optimizing semantic importance and information freshness.

## Setup

    py -3.11 -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt

## Reproducing the results

    # 1. Verify the simulation
    python -m tests.test_environment

    # 2. Train the main model (~10 minutes)
    python -m src.train

    # 3. Evaluate against baselines on held-out seeds
    python -m src.final_eval

    # 4. Run the ablation study (~45 minutes)
    python -m src.ablation
    python -m src.ablation_eval

    # 5. Generate all figures
    python -m src.plot_results

## Key results

| Method | AoI ↓ | Utility ↑ | Fairness ↑ |
|---|---|---|---|
| Random | 5.34 | 109.34 | 0.943 |
| Round-Robin | 5.30 | 107.74 | 0.949 |
| AoI-based | 11.34 | 111.73 | 0.492 |
| Semantic-priority | 13.96 | 106.32 | 0.346 |
| **MAPPO (proposed)** | 8.95 | **175.64** | 0.672 |

See `outputs/figures/` for training curves, comparison chart,
radar plot, and ablation study.

## Directory layout

    configs/        YAML configuration
    src/            environment, MAPPO, baselines, evaluation
    tests/          unit tests
    outputs/        checkpoints, logs, figures