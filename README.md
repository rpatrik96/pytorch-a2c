# Advantage Actor Critic (A2C)
#### Written and _documented_ in PyTorch

This is a repository of the A2C reinforcement learning algorithm in the newest
PyTorch (as of 03.06.2019) including also Tensorboard logging. The `agent.py` 
file contains a wrapper around the neural network, which can come handy if
implementing e.g. [curiosity-driven exploration](https://github.com/pathak22/noreward-rl/tree/master/src).

Running should be straightforward, all the command line arguments can be found in
`utils.py`.  Running 
```bash
python ./main.py
```
should launch the training on Pong.

While trying to immerse into deep reinforcement learning, I created
this repo to give you a well documented resource for A2C, as 
in my opinion, most publicly available repositories are either
not self-explanatory or just not documented well.

I would like to list the repos I collected lots of help,
please not that many of them offer a much wider range of functionality,
which was not the goal in my case.
- https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
- https://github.com/ikostrikov/pytorch-a3c
- https://github.com/lnpalmer/A2C 