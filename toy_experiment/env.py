import random

import gym


class GridWorld(gym.Env):
    def __init__(self, env_flag=2):
        super(gym.Env).__init__()
        # A, B, C, s_0, D
        # At s0, we can choose to go to (A, B, C) or D (2 dim. action)
        # At (A, B, C), we can only go to s0, under each action
        # At D, the trajectory terminates.
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = None
        self._grid_escape_time = 0
        self._grid_max_time = 100
        self._current_position = 0
        self.env_flag = env_flag
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
            1: 0,
            2: 10,
            3: -10,
            4: 5
        }
        self.state_space = len(self.reward_setting)

    def make_one_hot(self, state):
        vec = [0] * self.state_space
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
        elif self._current_position == 1:
            # keep at D
            next_state = 1
        elif self._current_position in self.middle_state:
            # to s0
            next_state = 0
        else:
            raise NotImplementedError('current position exceeds range!!!')
        done = next_state == 1
        if self._grid_escape_time > self._grid_max_time:
            done = True
        reward = self.reward_setting[next_state]
        info['current_position'] = self._ind_to_name[next_state]
        next_state_vector = self.make_one_hot(next_state)
        self._current_position = next_state
        return next_state_vector, reward, done, info

    def reset(self):
        self._grid_escape_time = 0
        self._current_position = 0
        state = self.make_one_hot(self._current_position)
        return state

    def seed(self, seed=None):
        super(gym.Env).seed(seed)
        self.action_space.seed(seed)


class RandomGridWorld(GridWorld):
    def __init__(self):
        self.possible_choice = [2, 3, 4]
        self.renv_flag = random.choice(self.possible_choice)
        super(RandomGridWorld, self).__init__(self.renv_flag)

    def reset(self):
        self.renv_flag = random.choice(self.possible_choice)
        self.env_flag = self.renv_flag
        return super(RandomGridWorld, self).reset()


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

