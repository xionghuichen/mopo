from rnn_base import RNNBase
import torch
from net_config import rnn_config, fc_config


class Policy:
    def __init__(self, state_dim, action_dim, use_fc=True):

        config = rnn_config if not use_fc else fc_config
        self.network = RNNBase(state_dim, action_dim, **config)
        self.act_dim = action_dim
        self.state_dim = state_dim

    def get_action(self, state, hidden):
        prediction, next_hidden = self.network.meta_forward(state, hidden)
        prediction = torch.softmax(prediction, dim=-1)
        return prediction, next_hidden

    def make_init_action(self, device=torch.device('cpu')):
        return torch.zeros((1, self.act_dim), device=device)

    def make_random_input(self, device=torch.device('cpu')):
        return torch.randn((1, self.state_dim), device=device)

    def make_init_hidden(self, batch_size=1, device=torch.device('cpu')):
        return self.network.make_init_state(batch_size, device)

    def parameter(self):
        return self.network.parameters(True)

    def to(self, *args, **kwargs):
        return self.network.to(*args, **kwargs)

    def save(self, path):
        self.network.save(path)

    def load(self, path, **kwargs):
        self.network.load(path, **kwargs)

if __name__ == '__main__':
    policy = Policy(5, 1, use_fc=False)
    hidden = policy.make_init_hidden(1)
    for i in range(10):
        with torch.no_grad():
            state = policy.make_random_input()
            action, hidden = policy.get_action(state, hidden)
            print(action, hidden)