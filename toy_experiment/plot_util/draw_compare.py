import matplotlib.pyplot as plt
from .plot_data import plot_res
import os
import numpy as np
import pandas as pd
from .plot_data import append_data
import json
# log_file的位置
data_root_path = '/Users/fanmingluo/Desktop/record'
# 图像输出的位置
fig_path = '/Users/fanmingluo/Desktop/record/pic/REDUCE_EP_LOSS_NEW'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
# 要画哪几个环境的
envs = ['Hopper', 'HalfCheetah', 'Walker2d']
# 要画哪几个具体方法的对比, 该方法在图中的图例叫什么名字
agent_short_name = {
    # 'OLD_BASELINE': None,
    #'FULL_MEMORY': None,
    #'REDUCE_FIXLENGTH': None,
    'REDUCE_EP_LOSS_NEW': 'Ours',
    # 'L2NORM_BOTTLE': 'Ours',
    'BASELINE': None
}
# 要对比这些方法的哪些方面，如非稳态的累计回报。key为对比的简称，value为x, y轴
xy_dict = {
    'non_stationary_return': ['TotalInteraction', 'EpRetNS'],
    'test_return': ['TotalInteraction', 'EpRetTest'],
    'Consistencyloss': ['TotalInteraction', 'ConsistencyLoss'],
    'DiversityLoss': ['TotalInteraction', 'DiverseLoss']

}

for k in agent_short_name:
    if agent_short_name[k] is None:
        agent_short_name[k] = k
for compare, xy_items in xy_dict.items():
    total_data_one_fig = {}
    for env in envs:
        data_dict = {k: [] for k in agent_short_name}
        ## ============================ collect data ============================
        # query path
        for method in agent_short_name:
            data_path = os.path.join(data_root_path, method)
            found_path = False
            data_path_lists = []
            for item in os.listdir(data_path):
                if item.startswith(env) and 'debug' not in item:
                    found_path = True
                    data_path_lists.append(os.path.join(data_path, item))
                    print(f'env: {env}, method: {method}, path: {data_path_lists[-1]}')
                    #break
            if not found_path:
                print('path not found!!!!')
                raise FileNotFoundError(f'log file not fouund!!!! in {data_path}')
            for _path in data_path_lists:
                # data = np.loadtxt(os.path.join(data_path, 'progress.txt'))
                data = pd.read_table(os.path.join(_path, 'progress.txt'))
                parameter_path = os.path.join(_path, 'parameter.json')
                parameter = json.load(open(parameter_path, 'r'))
                seed = parameter['seed']
                # print(f'parameter: {parameter}, seed: {seed}')
                x, y = map(lambda _data: np.array(_data.tolist()).reshape((-1, 1)), [data[xy_items[0]], data[xy_items[1]]])
                data_dict[method].append([x, y, seed])
        for k in data_dict:
            for x, y, seed in data_dict[k]:
                print(f'env: {env}, seed: {seed}, x_shape: {x.shape}, y_shape: {y.shape}, method: {k}, terms: {compare}')
        total_data_one_fig[env] = data_dict
    ## ============================ plot ============================
    # total_data_one_fig[env][method]
    # x, y, seed = total_data_one_fig['Hopper']['BASELINE']
    for env in envs:
        data_to_draw = []
        dict_to_draw = total_data_one_fig[env]
        for method in dict_to_draw:
            for x, y, seed in dict_to_draw[method]:
                data_to_draw = append_data(y, data_to_draw, label=agent_short_name[method], x_var=x)
        fig = plot_res(data_to_draw,
                 #shaded_std=False,
                 #shaded_err=True,
                 # figsize=None,
                 # figsize=(3.2, 2.24),
                 #figsize=(3.2, 2.5),
                 #legend_outside=False,
                 #resample=0,
                 #smooth_step=1,
                 xlabel=xy_items[0],
                 ylabel=xy_items[1]
                 )
        plt.xlim(left=0, right=1.5e6)
        plt.ticklabel_format(style='sci', scilimits=(-1,2), axis='x')
        fig[0].savefig(os.path.join(fig_path, f'{compare}-{env}.pdf'), bbox_inches='tight')
plt.show()