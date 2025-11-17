import numpy as np
import torch
from pufferlib import pufferl

env_name = "puffer_tetris"
args = pufferl.load_config(env_name)

# Load the trained model checkpoint
args["load_model_path"] = "latest"

# Disable rendering for headless evaluation
args["render_mode"] = "rgb_array"  # Use headless rendering mode
args["headless"] = True  # Force headless mode
args["fps"] = None  # Disable frame rate limiting
args["save_frames"] = 100  # Don't save frames
args["gif_path"] = "tetris_eval.mp4"  # Output video file path

# Use the pufferl.eval utility instead of creating a new experiment
print("Starting eval")
pufferl.eval(env_name, args)
print("Eval finished")
