import argparse


def get_params_from_file(filepath, params_name='params'):
    import importlib
    from dotmap import DotMap
    module = importlib.import_module(filepath)
    params = getattr(module, params_name)
    params = DotMap(params)
    return params


def get_variant_spec(command_line_args):
    from .base import get_variant_spec
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
from ray import tune

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

class ExperimentRunner(tune.Trainable):
    def _setup(self, variant):
        set_seed(variant['run_params']['seed'])

        self._variant = variant

        gpu_options = tf.GPUOptions(allow_growth=True)
        session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        tf.keras.backend.set_session(session)
        self._session = tf.keras.backend.get_session()

        self.train_generator = None
        self._built = False

    def _stop(self):
        tf.reset_default_graph()
        tf.keras.backend.clear_session()

    def _build(self):
        variant = copy.deepcopy(self._variant)

        environment_params = variant['environment_params']
        training_environment = self.training_environment = (
            get_environment_from_params(environment_params['training']))
        evaluation_environment = self.evaluation_environment = (
            get_environment_from_params(environment_params['evaluation'])
            if 'evaluation' in environment_params
            else training_environment)

        replay_pool = self.replay_pool = (
            get_replay_pool_from_variant(variant, training_environment))
        sampler = self.sampler = get_sampler_from_variant(variant)
        Qs = self.Qs = get_Q_function_from_variant(
            variant, training_environment)
        policy = self.policy = get_policy_from_variant(
            variant, training_environment, Qs)
        initial_exploration_policy = self.initial_exploration_policy = (
            get_policy('UniformPolicy', training_environment))

        #### get termination function
        domain = environment_params['training']['domain']
        static_fns = mopo.static[domain.lower()]
        ####

        self.algorithm = get_algorithm_from_variant(
            variant=self._variant,
            training_environment=training_environment,
            evaluation_environment=evaluation_environment,
            policy=policy,
            initial_exploration_policy=initial_exploration_policy,
            Qs=Qs,
            pool=replay_pool,
            static_fns=static_fns,
            sampler=sampler,
            session=self._session)

        initialize_tf_variables(self._session, only_uninitialized=True)

        self._built = True

    def _train(self):
        if not self._built:
            self._build()

        if self.train_generator is None:
            self.train_generator = self.algorithm.train()

        diagnostics = next(self.train_generator)

        return diagnostics

    def _pickle_path(self, checkpoint_dir):
        return os.path.join(checkpoint_dir, 'checkpoint.pkl')

    def _replay_pool_pickle_path(self, checkpoint_dir):
        return os.path.join(checkpoint_dir, 'replay_pool.pkl')

    def _tf_checkpoint_prefix(self, checkpoint_dir):
        return os.path.join(checkpoint_dir, 'checkpoint')

    def _get_tf_checkpoint(self):
        tf_checkpoint = tf.train.Checkpoint(**self.algorithm.tf_saveables)

        return tf_checkpoint

    @property
    def picklables(self):
        return {
            'variant': self._variant,
            'training_environment': self.training_environment,
            'evaluation_environment': self.evaluation_environment,
            'sampler': self.sampler,
            'algorithm': self.algorithm,
            'Qs': self.Qs,
            'policy_weights': self.policy.get_weights(),
        }

    def _save(self, checkpoint_dir):
        """Implements the checkpoint logic.

        TODO(hartikainen): This implementation is currently very hacky. Things
        that need to be fixed:
          - Figure out how serialize/save tf.keras.Model subclassing. The
            current implementation just dumps the weights in a pickle, which
            is not optimal.
          - Try to unify all the saving and loading into easily
            extendable/maintainable interfaces. Currently we use
            `tf.train.Checkpoint` and `pickle.dump` in very unorganized way
            which makes things not so usable.
        """
        pickle_path = self._pickle_path(checkpoint_dir)
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.picklables, f)

        if self._variant['run_params'].get('checkpoint_replay_pool', False):
            self._save_replay_pool(checkpoint_dir)

        tf_checkpoint = self._get_tf_checkpoint()

        tf_checkpoint.save(
            file_prefix=self._tf_checkpoint_prefix(checkpoint_dir),
            session=self._session)

        return os.path.join(checkpoint_dir, '')

    def _save_replay_pool(self, checkpoint_dir):
        replay_pool_pickle_path = self._replay_pool_pickle_path(
            checkpoint_dir)
        self.replay_pool.save_latest_experience(replay_pool_pickle_path)

    def _restore_replay_pool(self, current_checkpoint_dir):
        experiment_root = os.path.dirname(current_checkpoint_dir)

        experience_paths = [
            self._replay_pool_pickle_path(checkpoint_dir)
            for checkpoint_dir in sorted(glob.iglob(
                os.path.join(experiment_root, 'checkpoint_*')))
        ]

        for experience_path in experience_paths:
            self.replay_pool.load_experience(experience_path)

    def _restore(self, checkpoint_dir):
        assert isinstance(checkpoint_dir, str), checkpoint_dir

        checkpoint_dir = checkpoint_dir.rstrip('/')

        with self._session.as_default():
            pickle_path = self._pickle_path(checkpoint_dir)
            with open(pickle_path, 'rb') as f:
                picklable = pickle.load(f)

        training_environment = self.training_environment = picklable[
            'training_environment']
        evaluation_environment = self.evaluation_environment = picklable[
            'evaluation_environment']

        replay_pool = self.replay_pool = (
            get_replay_pool_from_variant(self._variant, training_environment))

        if self._variant['run_params'].get('checkpoint_replay_pool', False):
            self._restore_replay_pool(checkpoint_dir)

        sampler = self.sampler = picklable['sampler']
        Qs = self.Qs = picklable['Qs']
        # policy = self.policy = picklable['policy']
        policy = self.policy = (
            get_policy_from_variant(self._variant, training_environment, Qs))
        self.policy.set_weights(picklable['policy_weights'])
        initial_exploration_policy = self.initial_exploration_policy = (
            get_policy('UniformPolicy', training_environment))

        self.algorithm = get_algorithm_from_variant(
            variant=self._variant,
            training_environment=training_environment,
            evaluation_environment=evaluation_environment,
            policy=policy,
            initial_exploration_policy=initial_exploration_policy,
            Qs=Qs,
            pool=replay_pool,
            sampler=sampler,
            session=self._session)
        self.algorithm.__setstate__(picklable['algorithm'].__getstate__())

        tf_checkpoint = self._get_tf_checkpoint()
        status = tf_checkpoint.restore(tf.train.latest_checkpoint(
            os.path.split(self._tf_checkpoint_prefix(checkpoint_dir))[0]))

        status.assert_consumed().run_restore_ops(self._session)
        initialize_tf_variables(self._session, only_uninitialized=True)

        # TODO(hartikainen): target Qs should either be checkpointed or pickled.
        for Q, Q_target in zip(self.algorithm._Qs, self.algorithm._Q_targets):
            Q_target.set_weights(Q.get_weights())

        self._built = True


def get_trainable_class(*args, **kwargs):
    from .main import ExperimentRunner
    return ExperimentRunner

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


def main():
    import sys
    example_args = get_parser().parse_args(sys.argv[1:])
    variant_spec = get_variant_spec(example_args)
    trainable_class = ExperimentRunner(example_args)
    command_line_args = example_args
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

    variant = variant_spec
    # init
    set_seed(variant['run_params']['seed'])
    gpu_options = tf.GPUOptions(allow_growth=True)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.keras.backend.set_session(session)
    # build
    variant = copy.deepcopy(variant)

    environment_params = variant['environment_params']
    training_environment = (
        get_environment_from_params(environment_params['training']))
    evaluation_environment = (
        get_environment_from_params(environment_params['evaluation'])
        if 'evaluation' in environment_params
        else training_environment)

    replay_pool = (get_replay_pool_from_variant(variant, training_environment))
    sampler = get_sampler_from_variant(variant)
    Qs = get_Q_function_from_variant(variant, training_environment)
    policy = get_policy_from_variant(variant, training_environment, Qs)
    initial_exploration_policy = (get_policy('UniformPolicy', training_environment))

    #### get termination function
    domain = environment_params['training']['domain']
    static_fns = mopo.static[domain.lower()]
    ####

    algorithm = get_algorithm_from_variant(
        variant=variant,
        training_environment=training_environment,
        evaluation_environment=evaluation_environment,
        policy=policy,
        initial_exploration_policy=initial_exploration_policy,
        Qs=Qs,
        pool=replay_pool,
        static_fns=static_fns,
        sampler=sampler,
        session=session)

    initialize_tf_variables(session, only_uninitialized=True)
    # train
    algorithm.train()

if __name__=='__main__':
    main()