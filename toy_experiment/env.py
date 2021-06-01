import random

import gym


class GridWorld(gym.Env):
    def __init__(self, env_flag=2, append_context=False):
        super(gym.Env).__init__()
        # A, B, C, s_0, D
        # At s0, we can choose to go to (A, B, C) or D (2 dim. action)
        # At (A, B, C), we can only go to s0, under each action
        # At D, the trajectory terminates.
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = None
        self._grid_escape_time = 0
        self._grid_max_time = 10
        self._current_position = 0
        self.env_flag = env_flag
        self.append_context = append_context
        self.middle_state = [2, 3, 4]
        assert self.env_flag in self.middle_state, '{} is accepted.'.format(self.middle_state)
        self._ind_to_name = {
            0: 's0',
            1: 'D',
            2: 'A',
            3: 'B',
            4: 'C'
        }
        self.reward_setting = {
            0: 0,
            1: 1,
            2: 10,
            3: -5,
            4: 0
        }
        for k in self.reward_setting:
            self.reward_setting[k] *= 1.0

        self.state_space = len(self.reward_setting)
        self._raw_state_length = self.state_space
        if self.append_context:
            self.state_space += len(self.middle_state)

    @property
    def middle_state_embedding(self):
        v = [0] * len(self.middle_state)
        v[self.env_flag - 2] = 1
        return v

    def make_one_hot(self, state):
        vec = [0] * self._raw_state_length
        vec[state] = 1
        return vec

    def step(self, action):
        self._grid_escape_time += 1
        info = {}
        if self._current_position == 0:
            if action == 0:
                # to D
                next_state = 1
            else:
                # to unknown position
                next_state = self.env_flag
        # elif self._current_position == 1:
        #     # keep at D
        #     next_state = 1
        elif self._current_position in self.middle_state + [1]:
            # to s0
            next_state = 0
        else:
            raise NotImplementedError('current position exceeds range!!!')
        done = False # next_state == 1
        if self._grid_escape_time >= self._grid_max_time:
            done = True
        reward = self.reward_setting[next_state]
        info['current_position'] = self._ind_to_name[next_state]
        next_state_vector = self.make_one_hot(next_state)
        self._current_position = next_state
        if self.append_context:
            next_state_vector += self.middle_state_embedding
        return next_state_vector, reward, done, info

    def reset(self):
        self._grid_escape_time = 0
        self._current_position = 0
        state = self.make_one_hot(self._current_position)
        if self.append_context:
            state += self.middle_state_embedding
        return state

    def seed(self, seed=None):
        super(gym.Env).seed(seed)
        self.action_space.seed(seed)


class RandomGridWorld(GridWorld):
    def __init__(self, append_context=False):
        self.possible_choice = [2, 3, 4]
        self.renv_flag = random.choice(self.possible_choice)
        self.fix_env = None
        super(RandomGridWorld, self).__init__(self.renv_flag, append_context)

    def reset(self):
        if self.fix_env is None:
            self.renv_flag = random.choice(self.possible_choice)
            self.env_flag = self.renv_flag
        else:
            self.renv_flag = self.env_flag = self.fix_env
        return super(RandomGridWorld, self).reset()

    def set_fix_env(self, fix_env):
        self.renv_flag = self.env_flag = self.fix_env = fix_env

if __name__ == '__main__':
    grid_world = RandomGridWorld()
    action_space = grid_world.action_space
    grid_world.reset()
    ret = 0
    for i in range(100000):
        state, reward, done, info = grid_world.step(action_space.sample())
        ret += reward
        if done:
            print(grid_world.env_flag, grid_world._grid_escape_time, ret)
            state = grid_world.reset()
            ret = 0

