import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tests.test_environment import test_environment_runs
from tests.test_aoi import test_aoi_increases_and_resets
from tests.test_scheduler import test_capacity_respected

if __name__ == "__main__":
    print("1. Environment test")
    test_environment_runs()
    print("\n2. AoI test")
    test_aoi_increases_and_resets()
    print("\n3. Scheduler test")
    test_capacity_respected()