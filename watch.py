from pufferlib import pufferl

env_name = "puffer_tetris"
args = pufferl.load_config(env_name)
args["load_model_path"] = "latest"
vecenv = pufferl.load_env(env_name, args)
pufferl.eval("puffer_tetris")
