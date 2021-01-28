import numpy as np
import tensorflow as tf
from mopo.models.constructor import construct_model, format_samples_for_training
from RLA.easy_log.tester import tester
import gym
import os
import d4rl


def model_name(args):
    name = "{}-{}".format(args.env, args.quality)
    if args.separate_mean_var:
        name += '_smv'
    name += '_{}'.format(args.seed)
    return name

def get_package_path():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main(args):
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    tester.configure(task_name='model_learn', private_config_path=os.path.join(get_package_path(), 'rla_config_model_learn.yaml'),
                     run_file='train_model_offline.py', log_root=get_package_path())
    tester.log_files_gen()
    tester.print_args()

    env = gym.make('{}-{}-v0'.format(args.env, args.quality))
    dataset = d4rl.qlearning_dataset(env) # env.qlearning_dataset()
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]

    model = construct_model(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=args.hidden_dim,
                            num_networks=args.num_networks, num_elites=args.num_elites,
                            model_type=args.model_type, separate_mean_var=args.separate_mean_var,
                            name=model_name(args))

    dataset['rewards'] = np.expand_dims(dataset['rewards'], 1)
    train_inputs, train_outputs = format_samples_for_training(dataset)
    model.train(train_inputs, train_outputs,
                batch_size=args.batch_size, holdout_ratio=args.holdout_ratio, max_epochs=args.max_epochs, max_t=args.max_t)
    model.save(args.model_dir, args.num_networks)

# python mopo/scripts/train_model_offline.py --num-networks 7 --separate-mean-var --env walker2d --quality random
# python mopo/scripts/train_model_offline.py --num-networks 7 --env walker2d --quality medium-expert
# python mopo/scripts/train_model_offline.py --num-networks 50 --env halfcheetah --quality medium-replay --separate-mean-var
# docker run --rm -it --shm-size 50gb -v /home/luofm/Code/mopo:/root/mopo -v /home/luofm/.d4rl:/root/.d4rl sanluosizhou/selfdl:mopo -c "export CUDA_VISIBLE_DEVICES=1 && cd /root/mopo && pip install -e . && python mopo/scripts/train_model_offline.py --num-networks 7 --separate-mean-var --env walker2d --quality mixed"

if __name__ == '__main__':
    from argparse import ArgumentParser
    import os
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    parser = ArgumentParser()
    parser.add_argument('--env', default="halfcheetah")
    parser.add_argument('--quality', default="medium-replay")
    parser.add_argument('--info', default="")

    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--model-type', default='mlp')
    parser.add_argument('--separate-mean-var', action='store_true')
    parser.add_argument('--num-networks', default=7, type=int)
    parser.add_argument('--num-elites', default=5, type=int)
    parser.add_argument('--hidden-dim', default=200, type=int)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--holdout-ratio', default=0.2, type=float)
    parser.add_argument('--max-epochs', default=None, type=int)
    parser.add_argument('--max-t', default=1e10, type=float)
    parser.add_argument('--model-dir', default=os.path.join(get_package_path(), 'models'))
    args = parser.parse_args()
    kwargs = vars(args)
    tester.set_hyper_param(**kwargs)
    tester.add_record_param(['info',
                             "seed",
                             "env",
                             "quality" ])
    main(parser.parse_args())