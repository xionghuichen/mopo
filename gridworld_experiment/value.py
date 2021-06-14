from rnn_base import RNNBase
import torch
from net_config import rnn_config, fc_config


class Value:
    def __init__(self, state_dim, action_dim, use_fc=True):

        config = rnn_config if not use_fc else fc_config
        self.network = RNNBase(state_dim, 1, **config)
        self.act_dim = action_dim
        self.state_dim = state_dim

    def get_value(self, state, hidden):
        prediction, next_hidden = self.network.meta_forward(state, hidden)
        return prediction, next_hidden

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
    policy = Value(5, 1, use_fc=False)
