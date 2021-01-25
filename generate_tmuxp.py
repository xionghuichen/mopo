import argparse
import yaml
import os


def get_base_path():
    return os.path.dirname(os.path.abspath(__file__))
config = {"session_name": "run-all-1126", "windows": []}
base_path = get_base_path()
docker_path = "/root/mopo"
path = docker_path
tb_port = 6010
user_name = 'amax'

docker_template = f'docker run --rm -it --shm-size 50gb -v {base_path}:{docker_path} -v /home/{user_name}/.d4rl:/root/.d4rl sanluosizhou/selfdl:mopo -c '
docker_template_port = f'docker run --rm -it --shm-size 50gb -v {base_path}:{docker_path} -p {tb_port}:6006 sanluosizhou/selfdl:mopo -c '

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
                "examples.config.d4rl.hopper_mixed",
                "examples.config.d4rl.hopper_medium_expert",
                "examples.config.d4rl.hopper_medium",
                "examples.config.d4rl.hopper_random",
               ],
    'use_adapt': [True],
    'info': [
        'origin_mixed',
        'origin_medium_expert',
        'origin_medium',
        'origin_random',
             ],
    'length': [None, None, None, None],
    'penalty_coeff': [None, None, None, None],
    'elite_num': [None],
    'model_suffix': [None]
}



exp_num = len(params['info'])

template = docker_template + '\"export CUDA_VISIBLE_DEVICES={0} && cd {1} && pip install -e . ' \
           '&& python simple_run/main.py {2}\"'
template2 = docker_template_port + '"sleep 25 && cd {0} && tensorboard --logdir=./log/policy_learn"'

for i in range(exp_num):
    device_ind = i % 2
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
print(template2.format(path))
yaml.dump(config, open("run_all.yaml", "w"), default_flow_style=False)
