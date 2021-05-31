# Toy Task

## Environment Descriptions

**State space**

- *s0*, the initial state. Each trajectory starts at this state.
- *D*, terminate state. Trajectories stop at this state.
- *A, B, C*, middle states. 

**Action space**

- True
- False

**Reward function**

When arriving at

- *s0* and *D*, 0 reward;
- *A*, 10 reward;
- *B*, -10 reward;
- *C*, 5 reward.

**Transition function**

At *s0*, if action is 
- True, go to one of *A, B, C*
- False, go to *D*

At *A, B, C*, no matter what current action is, go to *s0*.

The maximum rollout length is 100.
**Uncertainty**

It is uncertain what state we will enter after executing *True* at *s0*. 
We construct all possible environments, i.e. three environments. 
For example, at the first environment, agents will always arrive *A* after executing *True* at *s0*.

**Optimal policy**
For the environment that can arrive at *A, C*, the optimal policy is always choose *True* at *s0*.
However, in the environment that can reach *B*, the optimal policy is choosing *False* at *s0*.
## Learn

We try to learn an adaptable policy on the constructed environments.
At the beginning of each trajectory, we randomly sample an environment from the environment set. 
Then, the agent will sample a trajectory from the environment.
We optimize the policy to maximize the expected long term return for 100 iterations via PPO.

## Test
We test the learned policy on all environments.
In order to make a comparison, we also learn a policy without adaptable ability.
### Results
**w/o adaptable ability**
We present the return as well as the trajectory sampled by the agent.
```
env_id: 2 , rets:  510.0 num:  101 , aver len:  101.0
s0->A->s0->A->s0->...A->s0->A
env_id: 3 , rets:  -510.0 num:  101 , aver len:  101.0
s0->B->s0->B->s0->...B->s0->B
env_id: 4 , rets:  255.0 num:  101 , aver len:  101.0
s0->C->s0->C->s0->...C->s0->C
```

**w/ adaptable ability**
```
env_id: 2 , rets:  510.0 num:  101 , aver len:  101.0
s0->A->s0->A->s0->...A->s0->A
env_id: 3 , rets:  -10.0 num:  3 , aver len:  3.0
s0->B->s0->D
env_id: 4 , rets:  255.0 num:  101 , aver len:  101.0
s0->C->s0->C->s0->...C->s0->C
```

From the behaviour of the agent in the env_id_3 environment, we can observe the differences between the agents.
The adaptable agent recognizes (probes) the environment after reaching state *B*, and goes to *D* subsequently.
The agent without adaptable ability only keeps a single behaviour pattern at the three environments.

We can obviously find a probing phase in the behaviour pattern of the adaptable policy:
The policy we try to choose *True* to probe the environment.
The environment can be determined after arriving at *A, B, C*.
Finally, the adaptable policy queries the optimal policy according to the environment.
## Minors

### network architecture
Refer to [net_config.py](net_config.py) for both fc network and rnn network.

### environment setting
Refer to [env.py](env.py) for environment setting.

### hyperparameter of PPO
Refer to [ppo.py](ppo.py) for ppo hyperparameter.

### conduct it
[run.py](run.py)


