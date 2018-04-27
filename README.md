# Hindsight Experience Replay and Hierarchical Reinforcement Learning
Comp 781 Project
https://github.com/ArmaanSethi/Comp-781-Project


## How to use Hindsight Experience Replay and Hierarchical Reinforcement Learning
Setup using OpenAI gyms. There are many tutorials online that will do a lot better than me.
Then you can clone this repo and use the code I added to the baselines they provided. 

### Getting started
Training an agent is very simple:
```bash
python -m baselines.herhrl.experiment.train
```
This will train a DDPG+HER+HRL agent on the `FetchReach` environment.
You should see the success rate go up quickly to `1.0`, which means that the agent achieves the
desired goal in all of the cases.
The training script logs other diagnostics as well and pickles the best policy so far (w.r.t. to its test success rate),
the latest policy, and, if enabled, a history of policies every K epochs.
Use the flag --env_name FetchPickAndPlace-v0
to change the environment.

To inspect what the agent has learned, use the play script:
```bash
python -m baselines.herhrl.experiment.play /path/to/an/experiment/policy_best.pkl
```
You can try it right now with the results of the training step (the script prints out the path for you).
This should visualize the current policy for 10 episodes and will also print statistics.

I used
```bash
python -m baselines.herhrl.experiment.train.py --num_cpu 2 --env_name FetchPush-v0 --n_epochs 200 --replay_strategy future
```

### Videos
https://youtu.be/pPzTOkPKF2o
https://youtu.be/PcBb0IYE4F0
https://youtu.be/7k19-bpJLTA

### Code
I initially created my own implementation of DDPG, HER, and added HRL to it. 
In order to evaluate it fairly I decided to use the baseline HER as the foundation to my method, and then adding changes to various places in order to implement HRL as well. 
This allowed me to use their logger, which was very helpful in creating the graphs. 

The code I modified are in baselines/baselines/herhrl/

More specifically I modified replay_buffer.py, rollout.py, her.py, ddpg.py, and actor_critic.py.
