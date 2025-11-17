import os

from pufferlib import pufferl

env_name = "puffer_tetris"
args = pufferl.load_config(env_name)

# Limit workers to available CPU cores
cpu_count = os.cpu_count() or 8
args["sweep"]["vec"]["num_envs"]["max"] = cpu_count

# For quick testing: reduce max_runs (default is 200) and training time
# args["max_runs"] = 2
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

# Run hyperparameter sweep using Protein algorithm with defaults
print(f"Starting sweep with {args['max_runs']} runs using Protein algorithm")
print(f"Optimizing metric: {args['sweep']['metric']}")
pufferl.sweep(args, env_name)

# After sweep completes, train with the best hyperparameters found
# The sweep automatically updates args with the best config
print("\nSweep complete. Starting full training with best hyperparameters...")
vecenv = pufferl.load_env(env_name, args)
policy = pufferl.load_policy(args, vecenv)
train_config = dict(**args["train"], env=env_name)
trainer = pufferl.PuffeRL(train_config, vecenv, policy)

while trainer.epoch < trainer.total_epochs:
    trainer.evaluate()
    logs = trainer.train()

trainer.print_dashboard()
trainer.close()
