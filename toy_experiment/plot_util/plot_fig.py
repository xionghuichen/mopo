import matplotlib.pyplot as plt
from plot_data import plot_res
import os
import numpy as np
import pandas as pd
from plot_data import append_data
import json


def main():
    log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'log')
    types = ['mlp', 'rnn']
    label_map = {
        'mlp': 'conservative',
        # 'mlp_context': 'oracle',
        'rnn': 'adaptable'
    }
    seeds = [0, 1, 2, 3, 4 , 5, 6, 7, 8, 9]
    x_data_name = 'iteration'
    x_data_label = x_data_name
    y_data_label = 'return'
    for y_data_name in ['Ret', 'ret_test_env_2', 'ret_test_env_3', 'ret_test_env_4']:
        data_to_draw = []

        for t in types:
            for s in seeds:
                progress_path = os.path.join(log_path, '{}_{}'.format(t, s), 'progress.txt')
                data = pd.read_table(progress_path)
                x = np.array(data[x_data_name].tolist()).reshape((-1, 1))
                y = np.array(data[y_data_name].tolist()).reshape((-1, 1))
                data_to_draw = append_data(y, data_to_draw, label=label_map[t], x_var=x)
        fig = plot_res(data_to_draw,
                       shaded_std=False,
                       shaded_err=True,
                       xlabel=x_data_label,
                       ylabel=y_data_label,
                       figsize=(3.2 * 1.5, 2.5 * 1.5),
                       )
        if y_data_name == 'Ret':
            plt.plot([0, 100], [20, 20], 'r-.', label='optimal')
            plt.plot([0, 100], [16, 16], 'k--', label='probe-reduce')
            plt.legend(['adaptable', 'conservative', 'optimal', 'probe-reduce'])

        plt.xlim(left=0, right=100)

        fig[0].savefig(os.path.join(log_path, f'compare_{y_data_name}.pdf'), bbox_inches='tight')

    rewards_conservative = [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]

    rewards_optimal = [
        [10, 10, 10, 10, 10],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]

    rewards_adaptable = [
        [10, 10, 10, 10, 10],
        [-10, 1, 1, 1, 1],
        [0, 1, 1, 1, 1]
    ]
    def add_zero(rewards):
        for i in range(len(rewards)):
            rewards[i] = [0] + rewards[i]
        return rewards
    length = 6
    rewards_conservative, rewards_optimal, rewards_adaptable = map(add_zero, [rewards_conservative, rewards_optimal, rewards_adaptable])
    for ind, (r_conservative, r_optimal, r_adaptable) in enumerate(zip(rewards_conservative,
                                                                       rewards_optimal,
                                                                       rewards_adaptable)):
        data_to_draw = []
        r_conservative, r_optimal, r_adaptable = map(lambda x: np.array(x).reshape((-1, 1)), [r_conservative, r_optimal, r_adaptable])
        print(ind, r_adaptable)
        # for i in range(3):
        #     data_to_draw = append_data(r_adaptable, data_to_draw, label='adaptable',
        #                                x_var=np.array([i for i in range(5)]).reshape((-1, 1)))
        #     data_to_draw = append_data(r_conservative, data_to_draw, label='conservative',
        #                                x_var=np.array([i for i in range(5)]).reshape((-1, 1)))
        #     data_to_draw = append_data(r_optimal, data_to_draw, label='optimal',
        #                                x_var=np.array([i for i in range(5)]).reshape((-1, 1)))
        # fig = plot_res(data_to_draw,
        #                # shaded_std=False,
        #                shaded_err=True,
        #                xlabel='step',
        #                ylabel='reward',
        #                figsize=(3.2 * 1.5, 2.5 * 1.5),
        #                )
        plt.subplots(1, 1, figsize=(3.2 * 1.3, 2.5 * 1.3))
        plt.cla()
        plt.plot(np.array([i for i in range(length)]).reshape((-1, 1)), r_adaptable, '-*', label='adaptable')
        plt.plot(np.array([i for i in range(length)]).reshape((-1, 1)), r_conservative, '-+', label='conservative')
        plt.plot(np.array([i for i in range(length)]).reshape((-1, 1)), r_optimal, '-x', label='optimal')
        plt.xlim(left=0, right=length-1)
        plt.xlabel('step')
        plt.ylabel('reward')
        plt.legend()

        plt.savefig(os.path.join(log_path, f'reward_{ind}.pdf'), bbox_inches='tight')


if __name__ == '__main__':
    main()