import argparse
import yaml
import os


def get_base_path():
    return os.path.dirname(os.path.abspath(__file__))
config = {"session_name": "run-all-1199", "windows": []}
base_path = get_base_path()
docker_path = "/root/mopo"
path = docker_path
tb_port = 6010
user_name = 'amax'

docker_template = f'docker run --rm -it --shm-size 50gb -v {base_path}:{docker_path}  -v /home/{user_name}/.d4rl:/root/.d4rl sanluosizhou/selfdl:mopo -c '
docker_template_port = f'docker run --rm -it --shm-size 50gb -v {base_path}:{docker_path} -p {tb_port}:6006 sanluosizhou/selfdl:mopo -c '
DEVICES = [1]
params = {
    'config': [
                "examples.config.d4rl.halfcheetah_mixed",
                "examples.config.d4rl.halfcheetah_medium_expert",
                "examples.config.d4rl.halfcheetah_medium",
                "examples.config.d4rl.halfcheetah_random",
                "examples.config.d4rl.halfcheetah_mixed",
                "examples.config.d4rl.halfcheetah_mixed",
                "examples.config.d4rl.halfcheetah_mixed",
                "examples.config.d4rl.halfcheetah_mixed",
               ],
    'use_adapt': [True],
    'info': [
        'origin_mixed',
        'origin_medium_expert',
        'origin_medium',
        'origin_random',
        'rollout_10_mixed',
        'rollout_15_mixed',
        'penalty_0.5_mixed',
        'penalty_0.25_mixed'
             ],
    'length': [None, None, None, None, 10, 15, None, None],
    'penalty_coeff': [None, None, None, None, None, None, 0.5, 0.25],
    'elite_num': [None],
    'model_suffix': [None]
}

params = {
    'config': [
                "examples.config.d4rl.walker2d_medium_expert",
                "examples.config.d4rl.walker2d_medium",
                "examples.config.d4rl.walker2d_random",
                "examples.config.d4rl.walker2d_medium",
                "examples.config.d4rl.hopper_medium_expert",
                "examples.config.d4rl.hopper_medium",
                "examples.config.d4rl.hopper_random",
                "examples.config.d4rl.hopper_mixed",
               ],
    'use_adapt': [True],
    'retrain_model': [True],
    'seed': [11,21,31,41,51,61,71,81],
    'info': [
        'walker2d_medium_expert',
        'walker2d_medium',
        'walker2d_random',
        'walker2d_mixed',
        'hopper_medium_expert',
        'hopper_medium',
        'hopper_random',
        'hopper_mixed',
             ],
    'length': [None],
    'penalty_coeff': [None],
    'elite_num': [None],
    'model_suffix': [None]
}


params = {
    'config': [
                "examples.config.d4rl.hopper_medium_expert",
                "examples.config.d4rl.hopper_medium",
                "examples.config.d4rl.hopper_random",
                "examples.config.d4rl.hopper_mixed",
                "examples.config.d4rl.hopper_medium_expert",
                "examples.config.d4rl.hopper_medium",
                "examples.config.d4rl.hopper_random",
                "examples.config.d4rl.hopper_mixed",
               ],
    'use_adapt': [True],
    'retrain_model': [True],
    'seed': [11,21,31,41,51,61,71,81],
    'info': [
        'hopper_medium_expert_5_length',
        'hopper_medium_5_length',
        'hopper_random_5_length',
        'hopper_mixed_5_length',
        'hopper_medium_expert_0.25_penalty_5_length',
        'hopper_medium_0.25_penalty_5_length',
        'hopper_random_0.25_penalty_5_length',
        'hopper_mixed_0.25_penalty_5_length',
             ],
    'length': [5],
    'penalty_coeff': [None, None, None, None, 0.25, 0.25, 0.25, 0.25],
    'elite_num': [None],
    'model_suffix': [None]
}

params = {
    'config': [
        "examples.config.d4rl.hopper_mixed",
        "examples.config.d4rl.hopper_medium_expert",
        "examples.config.d4rl.hopper_medium",
        "examples.config.d4rl.hopper_random",
    ],
    "model_suffix": [50],
    "info": ['hopper_mixed',
            'hopper_medium_expert',
             'hopper_medium',
             'hopper_random',
             # 'halfcheetah'
             ],
    'penalty_coeff': [0.05],
    'length': [10],
    'use_adapt': [True],
    'seed': [8, 88, 888, 888],
    'retrain': [True]
}

params = {
    'config': [
        # "examples.config.d4rl.hopper_mixed",
        # "examples.config.d4rl.hopper_medium_expert",
        "examples.config.d4rl.walker2d_medium",
        # "examples.config.d4rl.hopper_random",
    ],
    "model_suffix": [20],
    "info": ['walker_debug_no_clip_large_pool_normal_pool_123',
            # 'hopper_medium_expert',
            #  'hopper_medium',
            #  'hopper_random',
             # 'halfcheetah'
             ],
    'penalty_coeff': [0.25],
    'length': [5],
    'use_adapt': [True],
    'seed': [888],
    # 'retrain': [True]
}

#
# params = {
#     'config': [
#                 "examples.config.d4rl.walker2d_medium",
#                ],
#     'use_adapt': [False],
#     'info': [
#         'medium_fc',
#              ],
#     'length': [None],
#     'penalty_coeff': [None],
#     'elite_num': [None],
#     'model_suffix': [None]
# }

exp_num = len(params['info'])

template = docker_template + '\"export CUDA_VISIBLE_DEVICES={0} && cd {1} && pip install -e . ' \
           '&& python simple_run/main.py {2}\"'
template2 = docker_template_port + '"sleep 25 && cd {0} && tensorboard --logdir=./log"'

template3 = docker_template + '\"cd {0} && pip install -e . ' \
           '&& python obs_mem_percent.py\"'

for i in range(exp_num):
    device_ind = DEVICES[i % len(DEVICES)]
    panes_list = []
    params_str = ''
    for k, v in params.items():
        value = v[0] if len(v) == 1 else v[i]
        if isinstance(value, bool):
            if value:
                params_str += ' --{} '.format(k)
        elif value is None:
            pass
        else:
            params_str += ' --{} {} '.format(k, value)
    script = template.format(device_ind, docker_path, params_str)
    panes_list.append(
        script
    )
    print('info: {}, cmd: {}'.format(params['info'][i], script))
    config["windows"].append({
        "window_name": "{}".format(params['info'][i]),
        "panes": panes_list,
        "layout": "tiled"
    })

config["windows"].append({
        "window_name": "tensorboard",
        "panes": [template2.format(path)],
        "layout": "tiled"
    })
config["windows"].append({
        "window_name": "mem_percent",
        "panes": [template3.format(path)],
        "layout": "tiled"
    })

print(template2.format(path))
print(template3.format(path))
yaml.dump(config, open("run_all.yaml", "w"), default_flow_style=False)
