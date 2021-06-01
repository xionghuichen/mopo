import os.path

import torch

from env import RandomGridWorld
from policy import Policy
from policy_continuous import PolicyCont
from value import Value
from replay_memory import MemoryNp
import numpy as np
import gym

class PPO:
    def __init__(self, use_fc=False, use_context=False, discrete=True):
        self.use_fc = use_fc
        self.discrete = discrete
        self.use_context = use_context
        if self.discrete:
            self.env = RandomGridWorld(append_context=use_context)
            state_dim = self.env.state_space
        else:
            self.env = gym.make('HalfCheetah-v2')
            state_dim = self.env.observation_space.shape[0]

        print('env observation dim: {} env act dim: {}'.format(state_dim, self.env.action_space))
        policy_arch = Policy if self.discrete else PolicyCont
        self.policy = policy_arch(state_dim, 1 if self.discrete else self.env.action_space.shape[0], use_fc)
        self.value = Value(state_dim, 1, True)
        self.policy_optim = torch.optim.Adam(self.policy.network.parameters(True), lr=3e-4)
        self.value_optim = torch.optim.Adam(self.value.network.parameters(True), lr=1e-3)
        self.device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda:0')
        self.gamma = 0.98
        self.lam = 0.98
        self.clip_param = 0.1
        self.ent_coeff = 0.0
        self.policy.network.to(self.device)
        self.value.network.to(self.device)

    def get_gaes(self, traj):
        reward = [item[3] for item in traj]
        value = [item[5] for item in traj]
        done = [item[4] for item in traj]
        returns, gae = self._get_gae(reward, done, value)
        return gae, returns

    def _get_gae(self, rewards, done, values):
        returns = [0] * len(rewards)
        advants = [0] * len(rewards)
        masks = [1 - d for d in done]
        assert not masks[-1]
        running_returns = 0
        previous_value = 0
        running_advants = 0

        for t in reversed(range(0, len(rewards))):
            running_returns = rewards[t] + self.gamma * running_returns * masks[t]
            running_tderror = rewards[t] + self.gamma * previous_value * masks[t] - \
                              values[t]
            running_advants = running_tderror + self.gamma * self.lam * \
                              running_advants * masks[t]

            returns[t] = running_returns
            previous_value = values[t]
            advants[t] = running_advants
        # print([*zip(rewards, returns, advants, values)])
        # exit(0)
        return returns, advants

    def sample(self, traj_num, max_transition_num=20000, deterministic=False):
        memory = MemoryNp(rnn_slice_length=1)
        total_step = 0
        for traj_ind in range(traj_num):
            hidden = self.policy.make_init_hidden(1, device=self.device)
            value_hidden = self.value.make_init_hidden(1, device=self.device)
            state = self.env.reset()
            done = False
            traj = []
            while not done:
                state_tensor = torch.Tensor(state).to(device=self.device).unsqueeze(0)
                value, value_hidden = self.value.get_value(state_tensor, value_hidden)
                if self.discrete:
                    action_distribution, hidden = self.policy.get_action(state_tensor, hidden)
                    dist = torch.distributions.Bernoulli(action_distribution)
                    sampled_action = dist.sample()
                    logp = dist.log_prob(sampled_action)
                    action = sampled_action.item()
                    if deterministic:
                        action = 1 if action_distribution.item() > 0.5 else 0
                    action = action
                else:
                    action_mean, hidden = self.policy.get_action(state_tensor, hidden)
                    dist = torch.distributions.Normal(action_mean, self.policy.sigma * torch.ones_like(action_mean))
                    sampled_action = dist.sample()
                    logp = dist.log_prob(sampled_action).sum()
                    action = sampled_action.cpu().detach().numpy().reshape((-1))
                    if deterministic:
                        action = action_mean.cpu().detach().numpy().reshape((-1))
                    pass
                next_state, reward, done, info = self.env.step(action)
                if self.discrete:
                    action = [action]
                traj.append([state, action, next_state, reward, done, value.item(), logp.item()])
                state = next_state
                total_step += 1
            gae, returns = self.get_gaes(traj)
            for (s, a, s_next, r, d, v, logp), g, ret in zip(traj, gae, returns):
                memory.push(np.array(s), np.array(a), np.array([1]),
                            np.array(s_next), np.array([r]), np.array([g]),
                            np.array([v]), np.array([ret]), np.array([logp]), np.array([d]),
                            np.array([1]))
            if total_step > max_transition_num:
                break
        return memory

    def train(self, mem: MemoryNp, batch_size=1024, batch_num=80):
        actor_losses = []
        value_losses = []
        entropies = []
        # get mean and std of the gae
        gaes = []
        for t in mem.memory_buffer:
            for item in t:
                gaes.append(item.gae)
        gae_mean = np.mean(gaes)
        gae_std = np.std(gaes)
        returns = []
        for t in mem.memory_buffer:
            for item in t:
                returns.append(item.ret)
        return_mean = np.mean(returns)
        return_std = np.std(returns)
        for _ in range(batch_num):
            trajs, total_num = mem.sample_trajs(batch_size, 10000)
            state, action, gae, ret, mask, old_logp = trajs.state, trajs.action, trajs.gae, trajs.ret, trajs.mask, trajs.logp
            traj_num = state.shape[0]
            state, action, gae, ret, mask, old_logp = map(lambda x: torch.Tensor(x).to(device=self.device), [
                state, action, gae, ret, mask, old_logp
            ])

            valid_num = mask.sum().item()
            policy_hidden = self.policy.make_init_hidden(traj_num, device=self.device)
            value_hidden = self.value.make_init_hidden(traj_num, device=self.device)
            policy_out, _ = self.policy.get_action(state, policy_hidden)
            if self.discrete:
                distributions = torch.distributions.Bernoulli(policy_out)
            else:
                distributions = torch.distributions.Normal(policy_out, self.policy.sigma * torch.ones_like(policy_out))
            logp = distributions.log_prob(action)
            entropy = (distributions.entropy() * mask).sum() / valid_num
            entropies.append(entropy.item())
            ratio = torch.exp(logp - old_logp)
            gae = (gae - gae_mean) / gae_std
            surrogate = ratio * gae
            clipped_ratio = torch.clamp(ratio,
                                        1.0 - self.clip_param,
                                        1.0 + self.clip_param)
            clipped_loss = clipped_ratio * gae
            actor_loss = -(torch.min(surrogate, clipped_loss) * mask).sum() / valid_num
            actor_loss = actor_loss - self.ent_coeff * entropy
            self.policy_optim.zero_grad()

            actor_loss.backward()
            self.policy_optim.step()
            actor_losses.append(actor_loss.item())
            # value
            value_out, _ = self.value.get_value(state, value_hidden)
            value_loss = ((value_out - ret) * mask).pow(2).sum() / valid_num
            self.value_optim.zero_grad()
            value_loss.backward()
            self.value_optim.step()
            value_losses.append(value_loss.item())
            # policy_out = pol
        print('actor loss: {}, critic loss: {}, entropy: {}, '
              'gae_mean: {}, gae_std: {}, return_mean: {}, '
              'return std: {}'.format(np.mean(actor_losses),
                                       np.mean(value_losses),
                                       np.mean(entropies),
                                       gae_mean,
                                       gae_std,
                                       return_mean,
                                       return_std))

    def run(self):
        for iter in range(100):
            print('-'*15+str(iter)+'-'*15+'rnn:{}'.format(not self.use_fc))
            mem = self.sample(10000)
            print('rets: ', np.mean(mem.rets), 'num: ', mem.size, 'traj num:', len(mem.memory_buffer))
            self.train(mem, batch_size=4000, batch_num=50)
            if self.discrete:
                self.test(load=False)
            self.save()

    def get_trajectory_illustration(self, state, next_state):
        state = np.vstack([state, next_state[-1:]])
        state_str = ''
        for i in range(state.shape[0]):
            state_ = state[i]
            ind = 0
            for j in range(state.shape[1]):
                if state_[j]:
                    ind = j
                    break
            name = self.env._ind_to_name[ind]
            state_str += name
            if i < state.shape[0] - 1:
                state_str += '->'
        if len(state_str) > 20:
            state_str = state_str[:18] + '...' + state_str[-8:]
        return state_str

    def test(self, load=True):
        if load:
            self.load()
        for env in [2, 3, 4]:
            self.env.set_fix_env(env)
            mem = self.sample(1, 30000, True)
            trajs, transition_num = mem.sample_trajs(1)
            print('env_id: {}'.format(env), ', rets: ', np.mean(mem.rets), 'num: ', mem.size, ', aver len: ', mem.size / len(mem.memory_buffer))
            print(self.get_trajectory_illustration(trajs.state[0], trajs.next_state[0]))
        self.env.set_fix_env(None)

    @property
    def model_path(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_path, str(not self.use_fc).lower() + '_' + str(self.use_context), 'policy.pt')

        return path

    def save(self):
        path = self.model_path
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        self.policy.network.save(path)

    def load(self):
        path = self.model_path
        self.policy.network.load(path, map_location=self.device)

if __name__ == '__main__':
    ppo = PPO(False)
    ppo.run()