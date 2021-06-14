import argparse
import yaml
import os
# template = "vdfe &&sleep {0} && export CUDA_VISIBLE_DEVICES={1} && cd SLBDAO/dfe_sac/src && python main.py --seed {2} --env_id {4} --info {5} --anchor_state_size {6} --preserve_ratio {7} "
config = {"session_name": "run-all-8829", "windows": []}


def get_base_path():
    return os.path.dirname(os.path.abspath(__file__))

template = 'export CUDA_VISIBLE_DEVICES={0} && cd {1} ' \
           ' && vopen && python run.py --seed {2} '
seeds = [i for i in range(10)]
GPUS = [0, 1]
task_setting = [
    [True, False],
    [False, False],
    # [False, True]
]
# vrdc_update_interval = 100
imitate_update_interval = 50
gpu_ind = 0
group_num = 5
for group in range(len(seeds) // group_num):
    panes_list = []
    for use_rnn, use_context in task_setting:
        script_total = 'sleep 1 '
        for i in range(group_num):
            seed = seeds[i + group * group_num]
            script_it = template.format(GPUS[gpu_ind % len(GPUS)], get_base_path(), seed)
            if use_rnn:
                script_it += ' --adapt '
            if use_context:
                script_it += ' --context '
            script_total += '&& {}'.format(script_it)
        panes_list.append(script_total)
        print('seed: {}'.format(group), script_total)
    gpu_ind += 1
    config["windows"].append({
        "window_name": "seed:{}".format(seed),
        "panes": panes_list,
        "layout": "tiled"
    })

yaml.dump(config, open("run_all.yaml", "w"), default_flow_style=False)
