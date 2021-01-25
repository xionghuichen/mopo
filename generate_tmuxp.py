import argparse
import yaml
import os
def get_base_path():
    return os.path.dirname(os.path.abspath(__file__))
# template = "vdfe &&sleep {0} && export CUDA_VISIBLE_DEVICES={1} && cd SLBDAO/dfe_sac/src && python main.py --seed {2} --env_id {4} --info {5} --anchor_state_size {6} --preserve_ratio {7} "
config = {"session_name": "run-all-1116", "windows": []}
base_path = get_base_path()
docker_path = "/root/mopo"
path = docker_path
tb_port = 6010

docker_template = f'docker run --rm -it --shm-size 50gb -v {base_path}:{docker_path} sanluosizhou/selfdl:mopo -c '
docker_template_port = f'docker run --rm -it --shm-size 50gb -v {base_path}:{docker_path} -p 6006:{tb_port} sanluosizhou/selfdl:mopo -c '

template = 'export CUDA_VISIBLE_DEVICES={0} && cd {1} ' \
           '&& python main.py --env_name {3} --history_length {4} --seed {5} ' \
           ' --task_num {6} --num_threads {7} --vrdc_update_interval {9} --vrdc_ratio {10} ' \
           ' --imitate_update_interval {11} --mi_ratio {12} --max_iter_num {13} ' \
           '--varying_params {14} --rnn_slice_num {15} --rnn_fix_length {16} --test_task_num {17} ' \
           '--sac_mini_batch_size {18} --ep_dim {19}'
template2 = docker_template_port + '"sleep 25 && cd {0} && tensorboard --logdir=./log_file"'
template3 = docker_template + '"cd {0}"'
seeds = [2]
rnn_slice_num = 16
envs = ['HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2']
# envs = ['HalfCheetah-v2']
count_it = 0
algs = ['sac']
num_thread = 4
task_common_num = 40
common_hist_num = 0
mutual_information_ratio = 0.1
max_iter_num = 3000
uniform_sample_from_rb = False
rnn_fix_length = 32
test_task_num = 40
share_ep = False
ep_dim = 2
if rnn_fix_length == 0:
    sac_mini_batch_size = 64 * 1000
else:
    sac_mini_batch_size = 8 * 1000
varying_params = [
    ' gravity body_mass dof_damping ',
]


vrdc_ratio = 1.0   # hist,    task_num,                 param, VRDC, mutual_information, vrc_only, uposi, vrdc-update-interva;
hist_and_taks_num = [
                     # [common_hist_num, task_common_num, False, True, False, False, False, -1],  # value regression only
    # =========================host2
    #                  [common_hist_num, task_common_num, False, True, False, False, False, -1, False, 4, 0],  # value regression only
                     [common_hist_num, task_common_num, False, True, False, False, False, -1, False, 2, 32],  # value regression only
                    [common_hist_num, task_common_num, False, True, False, False, False, -1, False, 2, 0],  # value regression only
    # -------------------------server
    #                  [common_hist_num, task_common_num, False, False, False, False, False, -1, False],  # value regression only
    #                  [common_hist_num, task_common_num, False, False, False, False, False, -1, True],  # value regression only
    # ==========================
                     # [common_hist_num, task_common_num, False, False, False, False, False, 1],    # TC only

                     # [common_hist_num, task_common_num, False, True, True, False, False, 1],    # MI + TC
                     # [common_hist_num, task_common_num, False, False, False, False, False, -1],  # Stack sac
                     # [common_hist_num, task_common_num, False, False, False, False, True, -1],  # UPOSI
                     # [0, task_common_num, True, False, False, False, False, -1],                # GroundTruth
                     # [0, task_common_num, False, False, False, False, False, -1],               # Invariance policy (vanilla)
                     ]
# vrdc_update_interval = 100
imitate_update_interval = 50
for seed in seeds:
    for env in envs:
        panes_list = []
        for varying_param in varying_params:
            for alg in algs:
                for hist, task_num, use_true_parameter, use_vrdc, use_mutual, vrc_only, use_uposi, vrdc_update_interval, share_ep, ep_dim, rnn_fix_length in hist_and_taks_num:
                    if rnn_fix_length == 0:
                        sac_mini_batch_size = 64 * 1000
                    else:
                        sac_mini_batch_size = 8 * 1000
                    count_it = count_it + 1
                    script_it = template.format(count_it % 2, path, alg, env, hist, seed, task_num,
                                                num_thread, count_it*10, vrdc_update_interval, vrdc_ratio,
                                                imitate_update_interval, mutual_information_ratio, max_iter_num,
                                                varying_param, rnn_slice_num, rnn_fix_length, test_task_num,
                                                sac_mini_batch_size, ep_dim)
                    if use_true_parameter:
                        script_it += ' --use_true_parameter '
                    if use_vrdc:
                        script_it += ' --use_vrdc '
                    if use_mutual:
                        script_it += ' --use_mutual_information '
                    if vrc_only:
                        script_it += ' --vrc_only '
                    if use_uposi:
                        script_it += ' --use_uposi '
                    if uniform_sample_from_rb:
                        script_it += ' --uniform_sample '
                    if share_ep:
                        script_it += ' --share_ep '
                    script_it = docker_template + '"{}"'.format(script_it)
                    print('{}-{}'.format(env, seed), ': ', script_it)

                    panes_list.append(script_it)
        config["windows"].append({
            "window_name": "{}-{}".format(env, seed),
            "panes": panes_list,
            "layout": "tiled"
        })

config["windows"].append({
        "window_name": "tensorboard",
        "panes": [template2.format(path)],
        "layout": "tiled"
    })
print(template2.format(path))

config["windows"].append({
        "window_name": "init_operation",
        "panes": [template3.format(path)],
        "layout": "tiled"
    })
print(template3.format(path))
yaml.dump(config, open("run_all.yaml", "w"), default_flow_style=False)
