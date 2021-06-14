from rnn_base import RNNBase
import torch
from net_config import rnn_config, fc_config, ep_config, mlp_ep_config


class Policy:
    def __init__(self, state_dim, action_dim, use_fc=True):
        config = fc_config
        ep_config_it = ep_config if not use_fc else mlp_ep_config
        ep_dim = 2
        self.ep = RNNBase(state_dim, ep_dim, **ep_config_it)
        self.network = RNNBase(state_dim + ep_dim, action_dim, **config)
        self.act_dim = action_dim
        self.state_dim = state_dim

    def parameter(self):
        return [*self.ep.parameters(True)] + [*self.network.parameters(True)]

    def to(self, *args, **kwargs):
        self.ep.to(*args, **kwargs)
        return self.network.to(*args, **kwargs)

    def save(self, path):
        self.network.save(path)
        self.ep.save(path+'ep')

    def load(self, path, **kwargs):
        self.network.load(path, **kwargs)
        self.network.load(path+'ep', **kwargs)

    def get_action(self, state, hidden):
        ep, next_hidden = self.ep.meta_forward(state, hidden)
        prediction, _ = self.network.meta_forward(torch.cat((state, ep), -1), [])
        prediction = torch.softmax(prediction, dim=-1)
        return prediction, next_hidden

    def make_init_action(self, device=torch.device('cpu')):
        return torch.zeros((1, self.act_dim), device=device)

    def make_random_input(self, device=torch.device('cpu')):
        return torch.randn((1, self.state_dim), device=device)

    def make_init_hidden(self, batch_size=1, device=torch.device('cpu')):
        return self.ep.make_init_state(batch_size, device)


if __name__ == '__main__':
    policy = Policy(5, 1, use_fc=False)
    hidden = policy.make_init_hidden(1)
    for i in range(10):
        with torch.no_grad():
            state = policy.make_random_input()
            action, hidden = policy.get_action(state, hidden)
            print(action, hidden)