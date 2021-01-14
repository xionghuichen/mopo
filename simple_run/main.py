import sys
from RLA.easy_log.tester import tester
sys.path.append("../")

def get_params_from_file(filepath, params_name='params'):
    import importlib
    from dotmap import DotMap
    module = importlib.import_module(filepath)
    params = getattr(module, params_name)
    params = DotMap(params)
    return params


def get_variant_spec(command_line_args):
    from simple_run.base import get_variant_spec
    import importlib
    params = get_params_from_file(command_line_args.config)
    # import pdb
    # pdb.set_trace()
    variant_spec = get_variant_spec(command_line_args, params)
    return variant_spec


def get_parser():
    from examples.utils import get_parser
    parser = get_parser()
    return parser
import os


def add_command_line_args_to_variant_spec(variant_spec, command_line_args):
    variant_spec['run_params'].update({
        'checkpoint_frequency': (
            command_line_args.checkpoint_frequency
            if command_line_args.checkpoint_frequency is not None
            else variant_spec['run_params'].get('checkpoint_frequency', 0)
        ),
        'checkpoint_at_end': (
            command_line_args.checkpoint_at_end
            if command_line_args.checkpoint_at_end is not None
            else variant_spec['run_params'].get('checkpoint_at_end', True)
        ),
    })

    variant_spec['restore'] = command_line_args.restore

    return variant_spec

def generate_experiment(trainable_class, variant_spec, command_line_args):
    params = variant_spec.get('algorithm_params')
    local_dir = os.path.join(params.get('log_dir'), params.get('domain'))
    resources_per_trial = _normalize_trial_resources(
        command_line_args.resources_per_trial,
        command_line_args.trial_cpus,
        command_line_args.trial_gpus,
        command_line_args.trial_extra_cpus,
        command_line_args.trial_extra_gpus)

    experiment_id = params.get('exp_name')

    #### add pool_load_max_size to experiment_id
    if 'pool_load_max_size' in variant_spec['algorithm_params']['kwargs']:
        max_size = variant_spec['algorithm_params']['kwargs']['pool_load_max_size']
        experiment_id = '{}_{}e3'.format(experiment_id, int(max_size/1000))
    ####

    variant_spec = add_command_line_args_to_variant_spec(
        variant_spec, command_line_args)

    if command_line_args.video_save_frequency is not None:
        assert 'algorithm_params' in variant_spec
        variant_spec['algorithm_params']['kwargs']['video_save_frequency'] = (
            command_line_args.video_save_frequency)

    def create_trial_name_creator(trial_name_template=None):
        if not trial_name_template:
            return None

        def trial_name_creator(trial):
            return trial_name_template.format(trial=trial)

        return tune.function(trial_name_creator)

    experiment = {
        'run': trainable_class,
        'resources_per_trial': resources_per_trial,
        'config': variant_spec,
        'local_dir': local_dir,
        'num_samples': command_line_args.num_samples,
        'upload_dir': command_line_args.upload_dir,
        'checkpoint_freq': (
            variant_spec['run_params']['checkpoint_frequency']),
        'checkpoint_at_end': (
            variant_spec['run_params']['checkpoint_at_end']),
        'trial_name_creator': create_trial_name_creator(
            command_line_args.trial_name_template),
        'restore': command_line_args.restore,  # Defaults to None
    }

    return experiment_id, experiment

import sys
import pdb

import tensorflow as tf

import d4rl
from softlearning.environments.utils import get_environment_from_params
from softlearning.algorithms.utils import get_algorithm_from_variant
from softlearning.policies.utils import get_policy_from_variant, get_policy
from softlearning.replay_pools.utils import get_replay_pool_from_variant
from softlearning.samplers.utils import get_sampler_from_variant
from softlearning.value_functions.utils import get_Q_function_from_variant

from softlearning.misc.utils import set_seed, initialize_tf_variables
from examples.instrument import run_example_local
import pickle
import copy
import glob
import mopo.static



def _normalize_trial_resources(resources, cpu, gpu, extra_cpu, extra_gpu):
    if resources is None:
        resources = {}

    if cpu is not None:
        resources['cpu'] = cpu

    if gpu is not None:
        resources['gpu'] = gpu

    if extra_cpu is not None:
        resources['extra_cpu'] = extra_cpu

    if extra_gpu is not None:
        resources['extra_gpu'] = extra_gpu

    return resources

def get_package_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    import sys
    example_args = get_parser().parse_args(sys.argv[1:])

    variant_spec = get_variant_spec(example_args)
    command_line_args = example_args
    print('vriant spec: {}'.format(variant_spec))
    params = variant_spec.get('algorithm_params')
    local_dir = os.path.join(params.get('log_dir'), params.get('domain'))

    resources_per_trial = _normalize_trial_resources(
        command_line_args.resources_per_trial,
        command_line_args.trial_cpus,
        command_line_args.trial_gpus,
        command_line_args.trial_extra_cpus,
        command_line_args.trial_extra_gpus)
    experiment_id = params.get('exp_name')

    #### add pool_load_max_size to experiment_id
    if 'pool_load_max_size' in variant_spec['algorithm_params']['kwargs']:
        max_size = variant_spec['algorithm_params']['kwargs']['pool_load_max_size']
        experiment_id = '{}_{}e3'.format(experiment_id, int(max_size/1000))
    ####

    variant_spec = add_command_line_args_to_variant_spec(variant_spec, command_line_args)


    if command_line_args.video_save_frequency is not None:
        assert 'algorithm_params' in variant_spec
        variant_spec['algorithm_params']['kwargs']['video_save_frequency'] = (
            command_line_args.video_save_frequency)

    variant = variant_spec
    # init
    set_seed(variant['run_params']['seed'])
    gpu_options = tf.GPUOptions(allow_growth=True)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.keras.backend.set_session(session)

    # build
    variant = copy.deepcopy(variant)

    tester.set_hyper_param(**variant)
    tester.add_record_param(['run_params.seed'])
    tester.configure(task_name='policy_learn', private_config_path=os.path.join(get_package_path(), 'rla_config.yaml'),
                     run_file='main.py', log_root=get_package_path())
    tester.log_files_gen()
    tester.print_args()


    environment_params = variant['environment_params']
    training_environment = (get_environment_from_params(environment_params['training']))
    evaluation_environment = (get_environment_from_params(environment_params['evaluation'](variant))
        if 'evaluation' in environment_params else training_environment)

    replay_pool = (get_replay_pool_from_variant(variant, training_environment))
    sampler = get_sampler_from_variant(variant)
    Qs = get_Q_function_from_variant(variant, training_environment)
    policy = get_policy_from_variant(variant, training_environment, Qs)
    initial_exploration_policy = (get_policy('UniformPolicy', training_environment))

    #### get termination function
    domain = environment_params['training']['domain']
    static_fns = mopo.static[domain.lower()]
    ####
    print("[ DEBUG ] KWARGS: {}".format(variant['algorithm_params']['kwargs']))

    algorithm = get_algorithm_from_variant(
        variant=variant,
        training_environment=training_environment,
        evaluation_environment=evaluation_environment,
        policy=policy,
        initial_exploration_policy=initial_exploration_policy,
        Qs=Qs,
        pool=replay_pool,
        static_fns=static_fns,
        sampler=sampler, session=session)
    print('[ DEBUG ] finish construct model, start training')
    # train
    list(algorithm.train())

if __name__=='__main__':
    main()