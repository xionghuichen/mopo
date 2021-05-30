fc_config = {
            'hidden_size_list': [64, 32],
            'activation': ['tanh', 'tanh', 'linear'],
            'layer_type': ['fc', 'fc', 'fc']
        }
rnn_config = {
    'hidden_size_list': [64, 32],
    'activation': ['tanh', 'linear', 'linear'],
    'layer_type': ['fc', 'gru', 'fc']
}