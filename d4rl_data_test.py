import gym
import d4rl

if __name__ == '__main__':
    all_envs = ['hopper', 'walker2d', 'halfcheetah']
    all_tasks = [
        '-medium-v0',
        '-medium-replay-v0',
        '-medium-expert-v0',
        '-random-v0'
    ]
    for env in all_envs:
        for task in all_tasks:
            env_name = env + task
            print(env_name)
            data = d4rl.qlearning_dataset(gym.make(env_name))
            print(list(data.keys()), )
    pass