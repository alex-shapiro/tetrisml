from pufferlib import pufferl

env_name = "puffer_tetris"
args = pufferl.load_config(env_name)
vecenv = pufferl.load_env(env_name, args)
policy = pufferl.load_policy(args, vecenv)
train_config = dict(**args["train"], env=env_name)
trainer = pufferl.PuffeRL(train_config, vecenv, policy)

while trainer.epoch < trainer.total_epochs:
    trainer.evaluate()
    logs = trainer.train()

trainer.print_dashboard()
trainer.close()
