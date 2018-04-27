# Hindsight Experience Replay and Hierarchical Reinforcement Learning
For details on Hindsight Experience Replay (HER), please read the [paper](https://arxiv.org/abs/1707.01495).

## How to use Hindsight Experience Replay
Setup using OpenAI gyms. There are many tutorials online that will do a lot better than me.

### Getting started
Training an agent is very simple:
```bash
python -m baselines.herhrl.experiment.train
```
This will train a DDPG+HER+HRL agent on the `FetchReach` environment.
You should see the success rate go up quickly to `1.0`, which means that the agent achieves the
desired goal in 100% of the cases.
The training script logs other diagnostics as well and pickles the best policy so far (w.r.t. to its test success rate),
the latest policy, and, if enabled, a history of policies every K epochs.
Use the flag --env_name FetchPickAndPlace-v0
to change the environment.

To inspect what the agent has learned, use the play script:
```bash
python -m baselines.her.experiment.play /path/to/an/experiment/policy_best.pkl
```
You can try it right now with the results of the training step (the script prints out the path for you).
This should visualize the current policy for 10 episodes and will also print statistics.

I used
```bash
python -m baselines.her.experiment.train.py --num_cpu 2 --env_name FetchPush-v0 --n_epochs 200 --replay_strategy future
```

### Videos

https://youtu.be/pPzTOkPKF2o
https://youtu.be/PcBb0IYE4F0
https://youtu.be/7k19-bpJLTA

