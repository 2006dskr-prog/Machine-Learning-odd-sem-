"""
Reconstruct outputs/logs/history.npy from a completed training log.

Use this when training was interrupted (Ctrl+C) before the
final np.save() call at the end of train().
"""
import os
import numpy as np


# Values extracted from the terminal output of the training run
# Format: (episode, train_reward, eval_aoi, eval_pdr, eval_utility)
LOG_ENTRIES = [
    (0,    5647.82,  9.54,  0.41, 162.76),
    (100,  4953.17,  9.54,  0.41, 162.76),
    (200,  5042.72, 10.76,  0.41, 108.38),
    (300,  4368.06, 10.02,  0.41, 100.10),
    (400,  4657.14,  9.96,  0.42, 100.84),
    (500,  4703.31, 10.06,  0.41, 100.66),
    (600,  3508.84, 11.50,  0.41, 123.50),
    (700,  4037.13, 10.18,  0.43, 104.62),
    (800,  4382.30, 10.07,  0.41, 100.92),
    (900,  5018.23, 11.00,  0.42, 113.26),
    (1000, 3383.41, 11.88,  0.42, 123.04),
    (1100, 4991.21, 10.02,  0.41, 100.10),
    (1200, 4322.03, 12.80,  0.41, 132.76),
    (1300, 5357.06, 11.79,  0.41, 118.18),
    (1400, 4077.72, 12.10,  0.41, 121.86),
    (1500, 4850.68, 10.82,  0.42, 107.28),
]


def main():
    os.makedirs("outputs/logs", exist_ok=True)

    history = {
        'episode':      [e[0] for e in LOG_ENTRIES],
        'train_reward': [e[1] for e in LOG_ENTRIES],
        'eval_aoi':     [e[2] for e in LOG_ENTRIES],
        'eval_pdr':     [e[3] for e in LOG_ENTRIES],
        'eval_utility': [e[4] for e in LOG_ENTRIES],
    }

    out = "outputs/logs/history.npy"
    np.save(out, history)
    print(f"Saved {out}")
    print(f"  {len(history['episode'])} eval checkpoints reconstructed")
    print(f"  Episode range: {history['episode'][0]} – {history['episode'][-1]}")
    print(f"  AoI range: {min(history['eval_aoi']):.2f} – {max(history['eval_aoi']):.2f}")
    print(f"  Utility range: {min(history['eval_utility']):.2f} – {max(history['eval_utility']):.2f}")


if __name__ == "__main__":
    main()