import json
import os
from pathlib import Path

import psutil
from pufferlib import pufferl

env_name = "puffer_tetris"
args = pufferl.load_config(env_name)

# Limit workers and envs to available CPU cores
cpu_count = psutil.cpu_count(logical=False)
assert cpu_count is not None
print(f"CPU count: {cpu_count}")
args["sweep"]["vec"]["num_envs"]["max"] = cpu_count
args["vec"]["num_envs"] = cpu_count
args["vec"]["num_workers"] = cpu_count
args["vec"]["batch_size"] = cpu_count

# For quick testing: reduce max_runs (default is 200) and training time
args["max_runs"] = 50
# args["sweep"]["train"]["total_timesteps"]["min"] = 1_000_000  # 1M steps
# args["sweep"]["train"]["total_timesteps"]["max"] = 2_000_000  # 2M steps
# args["sweep"]["train"]["total_timesteps"]["mean"] = 1_500_000  # 1.5M steps

# Configure wandb
args["wandb"] = {
    "enabled": True,
    "entity": None,  # Set to your wandb username/team
    "project": "tetris-training",
    "group": "sweep-and-train",
}

# Check if sweep results already exist
sweep_config_path = Path("sweep_best_config.json")
if sweep_config_path.exists():
    print(f"Loading existing sweep results from {sweep_config_path}")
    with open(sweep_config_path, "r") as f:
        saved_config = json.load(f)
        # Update args with saved hyperparameters
        args.update(saved_config)
else:
    # Run hyperparameter sweep using Protein algorithm with defaults
    print(f"Starting sweep with {args['max_runs']} runs using Protein algorithm")
    print(f"Optimizing metric: {args['sweep']['metric']}")
    pufferl.sweep(args, env_name)

    # Save the best hyperparameters found by the sweep
    print(f"\nSweep complete. Saving best config to {sweep_config_path}")
    with open(sweep_config_path, "w") as f:
        json.dump(args, f, indent=2, default=str)

# Train with the best hyperparameters found
print("\nStarting full training with best hyperparameters...")
vecenv = pufferl.load_env(env_name, args)
policy = pufferl.load_policy(args, vecenv)
train_config = dict(**args["train"], env=env_name)
trainer = pufferl.PuffeRL(train_config, vecenv, policy)

while trainer.epoch < trainer.total_epochs:
    trainer.evaluate()
    logs = trainer.train()

trainer.print_dashboard()
trainer.close()
