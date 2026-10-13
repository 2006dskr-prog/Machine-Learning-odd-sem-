import numpy as np


class ChannelModel:
    """
    Distance-dependent probabilistic channel.

    Uses exponential decay with a reference distance so that vehicles
    a few hundred meters apart still have a reasonable success rate.
    """

    def __init__(self, success_prob_base: float = 0.95,
                 reference_distance: float = 500.0):
        self.base_prob = float(success_prob_base)
        self.ref_dist = float(reference_distance)

    def get_success_probability(self, distance: float) -> float:
        # At distance == ref_dist, prob = base * e^(-1) ≈ 0.37 * base
        # At distance == 0,     prob = base
        prob = self.base_prob * np.exp(-distance / self.ref_dist)
        return float(max(0.02, min(1.0, prob)))  # never fully zero

    def attempt_transmission(self, distance: float) -> bool:
        prob = self.get_success_probability(distance)
        return bool(np.random.rand() < prob)