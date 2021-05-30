import torch

from env import RandomGridWorld
from policy import Policy
from value import Value
from replay_memory import MemoryNp
import numpy as np
class PPO:
    def __init__(self, use_fc=False):
        self.use_fc = use_fc
        self.env = RandomGridWorld()
        self.policy = Policy(self.env.state_space, 1, use_fc)
        self.value = Value(self.env.state_space, 1, use_fc)
        self.policy_optim = torch.optim.Adam(self.policy.network.parameters(True), lr=3e-4)
        self.value_optim = torch.optim.Adam(self.value.network.parameters(True), lr=3e-4)
        self.device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda:0')
        self.gamma = 0.99
        self.lam = 0.97
        self.clip_param = 0.1

    def get_gaes(self, traj):
        reward = [item[3] for item in traj]
        value = [item[5] for item in traj]
        done = [item[4] for item in traj]
        gae, returns = self._get_gae(reward, done, value)
        return gae, returns

    def _get_gae(self, rewards, done, values):
        returns = [0] * len(rewards)
        advants = [0] * len(rewards)
        masks = [1 - d for d in done]

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
        return returns, advants

    def sample(self, traj_num, max_transition_num=20000):
        memory = MemoryNp(rnn_slice_length=1)
        total_step = 0
        for traj_ind in range(traj_num):
            hidden = self.policy.make_init_hidden(1, device=self.device)
            value_hidden = self.value.make_init_hidden(1, device=self.device)
            state = self.env.reset()
            done = False
            traj = []
            while not done:
                state_tensor = torch.Tensor(state, device=self.device).unsqueeze(0)
                action_distribution, hidden = self.policy.get_action(state_tensor, hidden)
                value, value_hidden = self.value.get_value(state_tensor, value_hidden)
                dist = torch.distributions.Bernoulli(action_distribution)
                sampled_action = dist.sample()
                logp = dist.log_prob(sampled_action)
                action = sampled_action.item()
                next_state, reward, done, info = self.env.step(action)
                traj.append([state, action, next_state, reward, done, value.item(), logp.item()])
                state = next_state
                total_step += 1
            gae, returns = self.get_gaes(traj)
            for (s, a, s_next, r, d, v, logp), g, ret in zip(traj, gae, returns):
                memory.push(np.array(s), np.array(a)[None], np.array([1]),
                            np.array(s_next), np.array([r]), np.array([g]),
                            np.array([v]), np.array([ret]), np.array([logp]), np.array([d]),
                            np.array([1]))
            if total_step > max_transition_num:
                break
        return memory

    def train(self, mem, batch_size=1024, batch_num=80):
        actor_losses = []
        value_losses = []
        entropies = []
        for _ in range(batch_num):
            trajs, total_num = mem.sample_trajs(batch_size, 10000)
            state, action, gae, ret, mask, old_logp = trajs.state, trajs.action, trajs.gae, trajs.ret, trajs.mask, trajs.logp
            traj_num = state.shape[0]
            state, action, gae, ret, mask, old_logp = map(lambda x: torch.Tensor(x, device=self.device), [
                state, action, gae, ret, mask, old_logp
            ])
            valid_num = mask.sum().item()
            policy_hidden = self.policy.make_init_hidden(traj_num, device=self.device)
            value_hidden = self.value.make_init_hidden(traj_num, device=self.device)
            policy_out, _ = self.policy.get_action(state, policy_hidden)
            distributions = torch.distributions.Bernoulli(policy_out)
            logp = distributions.log_prob(action)
            entropy = (distributions.entropy() * mask).sum() / valid_num
            entropies.append(entropy.item())
            ratio = torch.exp(logp - old_logp)
            surrogate = ratio * gae
            clipped_ratio = torch.clamp(ratio,
                                        1.0 - self.clip_param,
                                        1.0 + self.clip_param)
            clipped_loss = clipped_ratio * gae
            actor_loss = -(torch.min(surrogate, clipped_loss) * mask).sum() / valid_num
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
        print('actor loss: {}, critic loss: {}, entropy: {}'.format(np.mean(actor_losses), np.mean(value_losses), np.mean(entropies)))

    def run(self):
        for iter in range(100):
            print('-'*15+str(iter)+'-'*15+'rnn:{}'.format(not self.use_fc))
            mem = self.sample(1000)
            print('rets: ', np.mean(mem.rets), 'num: ', mem.size)
            self.train(mem)

if __name__ == '__main__':
    ppo = PPO(False)
    ppo.run()