from pufferlib import pufferl

env_name = "puffer_tetris"
args = pufferl.load_config(env_name)

# Load the trained model checkpoint
args["load_model_path"] = "latest"

# Enable rendering at human speed
args["render_mode"] = "human"  # Enable visual rendering
args["fps"] = 15  # Set frame rate (adjust for desired speed)
args["save_frames"] = 1  # Save frames for video/GIF
args["gif_path"] = "tetris_eval.gif"  # Output GIF file path

vecenv = pufferl.load_env(env_name, args)
policy = pufferl.load_policy(args, vecenv)
eval_config = dict(**args["train"], env=env_name)

rl = pufferl.PuffeRL(eval_config, vecenv, policy)
rl.evaluate()
rl.print_dashboard()
rl.close()
