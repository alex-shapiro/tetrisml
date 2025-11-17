import numpy as np
import pufferlib.pytorch
import torch
from pufferlib import pufferl

env_name = "puffer_tetris"
args = pufferl.load_config(env_name)

# Load the trained model checkpoint
args["load_model_path"] = "latest"

# Disable rendering completely for headless evaluation
args["env"]["render_mode"] = None

# Use single serial environment
args["vec"] = {"backend": "Serial", "num_envs": 1}

# Create environment and load trained policy
vecenv = pufferl.load_env(env_name, args)
policy = pufferl.load_policy(args, vecenv)
device = args["train"]["device"]

# Number of episodes to evaluate
num_episodes = 10
episode_returns = []

print(f"Running {num_episodes} evaluation episodes...")

for ep in range(num_episodes):
    ob, info = vecenv.reset()
    done = False
    episode_return = 0

    # Initialize RNN state if using recurrent policy
    # Get batch size from observation shape
    batch_size = ob.shape[0]
    state = {}
    if args["train"]["use_rnn"]:
        state["lstm_h"] = torch.zeros(batch_size, policy.hidden_size, device=device)
        state["lstm_c"] = torch.zeros(batch_size, policy.hidden_size, device=device)

    while not done:
        with torch.no_grad():
            ob_tensor = torch.as_tensor(ob).to(device)
            logits, value = policy.forward_eval(ob_tensor, state)
            action, _, _ = pufferlib.pytorch.sample_logits(logits)
            action = action.cpu().numpy().reshape(vecenv.action_space.shape)

        ob, reward, terminated, truncated, info = vecenv.step(action)
        episode_return += reward[0]
        done = terminated[0] or truncated[0]

    episode_returns.append(episode_return)
    print(f"  Episode {ep + 1}: return = {episode_return:.1f}")

vecenv.close()

# Print summary statistics
print(f"\n=== Evaluation Results ===")
print(f"Mean:   {np.mean(episode_returns):.2f}")
print(f"Median: {np.median(episode_returns):.2f}")
print(f"Std:    {np.std(episode_returns):.2f}")
print(f"Min:    {np.min(episode_returns):.2f}")
print(f"Max:    {np.max(episode_returns):.2f}")
