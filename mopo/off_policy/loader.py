import os
import glob
import pickle
import gzip
import pdb
import numpy as np

def restore_pool(replay_pool, experiment_root, max_size, save_path=None, adapt=False, maxlen=5):
    print('[ DEBUG ]: start restoring the pool')
    if 'd4rl' in experiment_root:
        restore_pool_d4rl(replay_pool, experiment_root[5:], adapt, maxlen)
    else:
        assert os.path.exists(experiment_root)
        if os.path.isdir(experiment_root):
            restore_pool_softlearning(replay_pool, experiment_root, max_size, save_path)
        else:
            try:
                restore_pool_contiguous(replay_pool, experiment_root)
            except:
                restore_pool_bear(replay_pool, experiment_root)
    print('[ mbpo/off_policy ] Replay pool has size: {}'.format(replay_pool.size))


def restore_pool_d4rl(replay_pool, name, adapt=False, maxlen=5):
    import gym
    import d4rl
    env = gym.make(name)
    mask_steps = env._max_episode_steps - 1
    data = d4rl.qlearning_dataset(gym.make(name))
    data['rewards'] = np.expand_dims(data['rewards'], axis=1)
    data['terminals'] = np.expand_dims(data['terminals'], axis=1)
    data['last_actions'] = np.concatenate((np.zeros((1, data['actions'].shape[1])), data['actions'][:-1, :]), axis=0).copy()
    # print(data['actions'] - data['last_actions'])
    data['first_step'] = np.zeros_like(data['terminals'])
    data['end_step'] = np.zeros_like(data['terminals'])
    data['valid'] = np.ones_like(data['terminals'])
    print('[ DEBUG ]: key in data: {}'.format(list(data.keys())))
    for i in range(data['observations'].shape[0]):
        flag = True
        if i >= 1:
            flag = (data['observations'][i] == data['next_observations'][i-1]).all()
        if not flag:
            data['last_actions'][i][:] = 0
            data['first_step'][i][:] = 1
            data['end_step'][i-1][:] = 1
    print("[ DEBUG ]: adapt is ", adapt)
    if adapt:

        data_adapt = {k: [] for k in data}
        it_traj = {k: [] for k in data}
        current_len = 0
        for start_ind in range(5):
            traj_start_ind = 0
            for i in range(data['observations'].shape[0]):
                if i - traj_start_ind < start_ind:
                    continue
                for k in data:
                    it_traj[k].append(data[k][i])
                current_len += 1
                if data['end_step'][i]:
                    traj_start_ind = i + 1
                    for j in range(maxlen - current_len):
                        for k in data:
                            it_traj[k].append(np.zeros_like(data[k][i]))
                        current_len += 1
                if current_len >= maxlen:
                    for k in data_adapt:
                        data_adapt[k].append(np.expand_dims(np.vstack(it_traj[k]), 0))
                    it_traj = {k: [] for k in data}
                    current_len = 0
        data_adapt = {k: np.vstack(v) for k, v in data_adapt.items()}
        data_adapt['last_actions'][:, 0] = 0
        for k, v in data_adapt.items():
            print('[ DEBUG ] key of env data: {}: value is {}'.format(k, v.shape))
        # print('[ DEBUG ] ----------')
        replay_pool.add_samples(data_adapt)
        return
    replay_pool.add_samples(data)


def restore_pool_softlearning(replay_pool, experiment_root, max_size, save_path=None):
    print('[ mopo/off_policy ] Loading SAC replay pool from: {}'.format(experiment_root))
    experience_paths = [
        checkpoint_dir
        for checkpoint_dir in sorted(glob.iglob(
            os.path.join(experiment_root, 'checkpoint_*')))
    ]

    checkpoint_epochs = [int(path.split('_')[-1]) for path in experience_paths]
    checkpoint_epochs = sorted(checkpoint_epochs)
    if max_size == 250e3:
        checkpoint_epochs = checkpoint_epochs[2:]

    for epoch in checkpoint_epochs:
        fullpath = os.path.join(experiment_root, 'checkpoint_{}'.format(epoch), 'replay_pool.pkl')
        print('[ mopo/off_policy ] Loading replay pool data: {}'.format(fullpath))
        replay_pool.load_experience(fullpath)
        if replay_pool.size >= max_size:
            break

    if save_path is not None:
        size = replay_pool.size
        stat_path = os.path.join(save_path, 'pool_stat_{}.pkl'.format(size))
        save_path = os.path.join(save_path, 'pool_{}.pkl'.format(size))
        d = {}
        for key in replay_pool.fields.keys():
            d[key] = replay_pool.fields[key][:size]

        num_paths = 0
        temp = 0
        path_end_idx = []
        for i in range(d['terminals'].shape[0]):
            if d['terminals'][i] or i - temp + 1 == 1000:
                num_paths += 1
                temp = i + 1
                path_end_idx.append(i)
        total_return = d['rewards'].sum()
        avg_return = total_return / num_paths
        buffer_max, buffer_min = -np.inf, np.inf
        path_return = 0.0
        for i in range(d['rewards'].shape[0]):
            path_return += d['rewards'][i]
            if i in path_end_idx:
                if path_return > buffer_max:
                    buffer_max = path_return
                if path_return < buffer_min:
                    buffer_min = path_return
                path_return = 0.0

        print('[ mopo/off_policy ] Replay pool average return is {}, buffer_max is {}, buffer_min is {}'.format(avg_return, buffer_max, buffer_min))
        d_stat = dict(avg_return=avg_return, buffer_max=buffer_max, buffer_min=buffer_min)
        pickle.dump(d_stat, open(stat_path, 'wb'))

        print('[ mopo/off_policy ] Saving replay pool to: {}'.format(save_path))
        pickle.dump(d, open(save_path, 'wb'))


def restore_pool_bear(replay_pool, load_path):
    print('[ mopo/off_policy ] Loading BEAR replay pool from: {}'.format(load_path))
    data = pickle.load(gzip.open(load_path, 'rb'))
    num_trajectories = data['terminals'].sum() or 1000
    avg_return = data['rewards'].sum() / num_trajectories
    print('[ mopo/off_policy ] {} trajectories | avg return: {}'.format(num_trajectories, avg_return))

    for key in ['log_pis', 'data_policy_mean', 'data_policy_logvar']:
        del data[key]

    replay_pool.add_samples(data)


def restore_pool_contiguous(replay_pool, load_path):
    print('[ mopo/off_policy ] Loading contiguous replay pool from: {}'.format(load_path))
    import numpy as np
    data = np.load(load_path)

    state_dim = replay_pool.fields['observations'].shape[1]
    action_dim = replay_pool.fields['actions'].shape[1]
    expected_dim = state_dim + action_dim + state_dim + 1 + 1
    actual_dim = data.shape[1]
    assert expected_dim == actual_dim, 'Expected {} dimensions, found {}'.format(expected_dim, actual_dim)

    dims = [state_dim, action_dim, state_dim, 1, 1]
    ends = []
    current_end = 0
    for d in dims:
        current_end += d
        ends.append(current_end)
    states, actions, next_states, rewards, dones = np.split(data, ends, axis=1)[:5]
    replay_pool.add_samples({
        'observations': states,
        'actions': actions,
        'next_observations': next_states,
        'rewards': rewards,
        'terminals': dones.astype(bool)
    })
