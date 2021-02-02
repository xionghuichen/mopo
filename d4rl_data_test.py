import gym
import d4rl
from d4rl.infos import DATASET_URLS
import os
import numpy as np

def get_base_path():
    return os.path.dirname(os.path.abspath(__file__))

def system(cmd):
    print(cmd)
    with os.popen(cmd, 'r') as f:
        for item in f:
            print(item)


def make_qlearning_dataset(data):
    data['next_observations'] = data['observations'][1:]
    data['next_observations'] = np.copy(np.concatenate((data['next_observations'], data['observations'][-1:]), axis=0))
    if not data['terminals'][-1]:
        print('final step is not terminal step, drop it')
        for k in data:
            data[k] = data[k][:-1]
    ind_to_delete = []
    for ind, (terminal, timeout) in enumerate(zip(data['terminals'], data['timeouts'])):
        if terminal:
            data['next_observations'][ind] = data['observations'][ind]
        if timeout:
            ind_to_delete.append(ind)
    for k in data:
        data[k] = np.delete(data[k], ind_to_delete, axis=0)
    return data

def main():
    all_envs = ['hopper', 'walker2d', 'halfcheetah']  # , 'ant']
    all_tasks = [
        '-medium-v0',
        '-medium-replay-v0',
        '-medium-expert-v0',
        '-random-v0',
        # '-expert-v0'
    ]
    need_download = False
    if need_download:
        if not os.path.exists(os.path.join(get_base_path(), 'datasets')):
            os.makedirs(os.path.join(get_base_path(), 'datasets'))
        system('rm -rf {}/*'.format(os.path.join(get_base_path(), 'datasets')))
    for env in all_envs:
        for task in all_tasks:
            env_name = env + task
            print(env_name)
            # data = d4rl.qlearning_dataset(gym.make(env_name))
            data = gym.make(env_name).get_dataset()
            data = make_qlearning_dataset(data)

            print(list(data.keys()), )
            for k in data:
                print('{}: {}'.format(k, data[k].shape))
            print('total timeouts: {}'.format(np.sum(data['timeouts'])))
            print('url: {}'.format(DATASET_URLS[env_name]))
            if need_download:
                system('cd {} && wget {}'.format(os.path.join(get_base_path(), 'datasets'), DATASET_URLS[env_name]))
    pass

if __name__ == '__main__':
    # data = make_qlearning_dataset(gym.make('hopper-medium-v0').get_dataset())
    data = gym.make('hopper-medium-expert-v0').get_dataset()
    for k in data:
        data[k] = data[k][309:]
    print(data['observations'][0])
    last_i = -1
    for i in range(data['observations'].shape[0]):

        if data['terminals'][i] or data['timeouts'][i]:
            print('{}, {}, done: {}, timeouts: {}'.format(i-last_i, i, data['terminals'][i], data['timeouts'][i]))
            if data['timeouts'][i]:
                print('='*20)
            if i - last_i > 1000:
                exit(1)
                print('='*40)
            last_i = i
