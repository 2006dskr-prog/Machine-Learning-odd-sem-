import numpy as np
from src.aoi_tracker import AoITracker


def test_aoi_increases_and_resets():
    tracker = AoITracker(num_vehicles=2)

    # no updates yet: AoI grows with time
    tracker.step(0.1)
    tracker.step(0.1)
    assert np.allclose(tracker.aoi_matrix, 0.2)

    # successful update from source 0 to receiver 1 at t=0.5
    tracker.update_aoi(source_id=0, receiver_id=1,
                       current_time=0.5, generation_time=0.3)
    assert np.isclose(tracker.aoi_matrix[0, 1], 0.2)

    # time advances
    tracker.step(0.1)
    assert np.isclose(tracker.aoi_matrix[0, 1], 0.3)
    print("AoI test passed.")


if __name__ == "__main__":
    test_aoi_increases_and_resets()