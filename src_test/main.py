import json
from execution.runner import run_admm_trajectory_optimization

CONFIG = "shape_3d_3"

with open(f"input/configs/{CONFIG}.json", "r") as f:
    config = json.load(f)

run_admm_trajectory_optimization(config, DEBUG=False)