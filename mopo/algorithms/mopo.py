## adapted from https://github.com/rail-berkeley/softlearning/blob/master/softlearning/algorithms/sac.py

import os
import math
import pickle
from collections import OrderedDict
from numbers import Number
from itertools import count
import gtimer as gt
import pdb

import numpy as np
import tensorflow as tf
from tensorflow.python.training import training_util

from softlearning.algorithms.rl_algorithm import RLAlgorithm
from softlearning.replay_pools.simple_replay_pool import SimpleReplayPool

from mopo.models.constructor import construct_model, format_samples_for_training
from mopo.models.fake_env import FakeEnv
from mopo.utils.writer import Writer
from mopo.utils.visualization import visualize_policy
from mopo.utils.logging import Progress
import mopo.utils.filesystem as filesystem
import mopo.off_policy.loader as loader
from RLA.easy_log.tester import tester
from d4rl import infos

def td_target(reward, discount, next_value):
    return reward + discount * next_value

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]


class MOPO(RLAlgorithm):
    """Model-based Offline Policy Optimization (MOPO)

    References
    ----------
        Tianhe Yu, Garrett Thomas, Lantao Yu, Stefano Ermon, James Zou, Sergey Levine, Chelsea Finn, Tengyu Ma. 
        MOPO: Model-based Offline Policy Optimization. 
        arXiv preprint arXiv:2005.13239. 2020.
    """

    def __init__(
            self,
            training_environment,
            evaluation_environment,
            policy,
            Qs,
            pool,
            static_fns,
            plotter=None,
            tf_summaries=False,
            lr=3e-4,
            reward_scale=1.0,
            target_entropy='auto',
            discount=0.99,
            tau=5e-3,
            target_update_interval=1,
            action_prior='uniform',
            reparameterize=False,
            store_extra_policy_info=False,
            adapt=False,
            gru_state_dim=256,
            network_kwargs=None,
            deterministic=False,
            rollout_random=False,
            model_train_freq=250,
            num_networks=7,
            num_elites=5,
            model_retain_epochs=20,
            rollout_batch_size=100e3,
            real_ratio=0.1,
            # rollout_schedule=[20,100,1,1],
            rollout_length=1,
            hidden_dim=200,
            max_model_t=None,
            model_type='mlp',
            separate_mean_var=False,
            identity_terminal=0,

            pool_load_path='',
            pool_load_max_size=0,
            model_name=None,
            model_load_dir=None,
            penalty_coeff=0.,
            penalty_learned_var=False,
            **kwargs):
        """
        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy: A policy function approximator.
            initial_exploration_policy: ('Policy'): A policy that we use
                for initial exploration which is not trained by the algorithm.
            Qs: Q-function approximators. The min of these
                approximators will be used. Usage of at least two Q-functions
                improves performance by reducing overestimation bias.
            pool (`PoolBase`): Replay pool to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.
            reparameterize ('bool'): If True, we use a gradient estimator for
                the policy derived using the reparameterization trick. We use
                a likelihood ratio based estimator otherwise.
        """

        super(MOPO, self).__init__(**kwargs)
        print("[ DEBUG ]: model name: {}".format(model_name))
        self._env_name = model_name[:-4] + '-v0'
        if self._env_name in infos.REF_MIN_SCORE:
            self.min_ret = infos.REF_MIN_SCORE[self._env_name]
            self.max_ret = infos.REF_MAX_SCORE[self._env_name]
        else:
            self.min_ret = self.max_ret = 0
        obs_dim = np.prod(training_environment.active_observation_shape)
        act_dim = np.prod(training_environment.action_space.shape)
        self._model_type = model_type
        self._identity_terminal = identity_terminal
        self._model = construct_model(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim,
                                      num_networks=num_networks, num_elites=num_elites,
                                      model_type=model_type, separate_mean_var=separate_mean_var,
                                      name=model_name, load_dir=model_load_dir, deterministic=deterministic)
        print('[ MOPO ]: got self._model')
        self._static_fns = static_fns
        self.fake_env = FakeEnv(self._model, self._static_fns, penalty_coeff=penalty_coeff,
                                penalty_learned_var=penalty_learned_var)

        self._rollout_schedule = [20, 100, rollout_length, rollout_length]
        self._max_model_t = max_model_t

        self._model_retain_epochs = model_retain_epochs

        self._model_train_freq = model_train_freq
        self._rollout_batch_size = int(rollout_batch_size)
        self._deterministic = deterministic
        self._rollout_random = rollout_random
        self._real_ratio = real_ratio
        # TODO: RLA writer (implemented with tf) should be compatible with the Writer object (implemented with tbx)
        self._log_dir = tester.log_dir
        # self._writer = tester.writer
        self._writer = Writer(self._log_dir)

        self._training_environment = training_environment
        self._evaluation_environment = evaluation_environment
        self.gru_state_dim = gru_state_dim
        self.network_kwargs = network_kwargs
        self.adapt = adapt
        self.optim_alpha = False
        # self._policy = policy

        # self._Qs = Qs
        # self._Q_targets = tuple(tf.keras.models.clone_model(Q) for Q in Qs)

        self._pool = pool
        self._plotter = plotter
        self._tf_summaries = tf_summaries

        self._policy_lr = lr
        self._Q_lr = lr

        self._reward_scale = reward_scale
        self._target_entropy = (
            -np.prod(self._training_environment.action_space.shape) * 1.5
            if target_entropy == 'auto'
            else target_entropy)
        print('[ MOPO ] Target entropy: {}'.format(self._target_entropy))

        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval
        self._action_prior = action_prior

        self._reparameterize = reparameterize
        self._store_extra_policy_info = store_extra_policy_info

        observation_shape = self._training_environment.active_observation_shape
        action_shape = self._training_environment.action_space.shape

        assert len(observation_shape) == 1, observation_shape
        self._observation_shape = observation_shape
        assert len(action_shape) == 1, action_shape
        self._action_shape = action_shape

        self._build()

        #### load replay pool data
        self._pool_load_path = pool_load_path
        self._pool_load_max_size = pool_load_max_size

        loader.restore_pool(self._pool, self._pool_load_path, self._pool_load_max_size, save_path=self._log_dir)
        self._init_pool_size = self._pool.size
        print('[ MOPO ] Starting with pool size: {}'.format(self._init_pool_size))
        ####

    def _build(self):

        self._training_ops = {}
        # place holder
        self.global_step = training_util.get_or_create_global_step()
        self._training_ops.update({
            'increment_global_step': training_util._increment_global_step(1)})

        self._iteration_ph = tf.placeholder(
            tf.int64, shape=None, name='iteration')

        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, None, *self._observation_shape),
            name='observation',
        )

        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, None, *self._observation_shape),
            name='next_observation',
        )

        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, None, *self._action_shape),
            name='actions',
        )

        self._prev_state_p_ph = tf.placeholder(
            tf.float32,
            shape=(None, self.gru_state_dim),
            name='prev_state_p',
        )
        self._prev_state_v_ph = tf.placeholder(
            tf.float32,
            shape=(None, self.gru_state_dim),
            name='prev_state_v',
        )

        self.seq_len = tf.placeholder(tf.float32, shape=[None], name="seq_len")

        self._rewards_ph = tf.placeholder(
            tf.float32,
            shape=(None, None, 1),
            name='rewards',
        )

        self._terminals_ph = tf.placeholder(
            tf.float32,
            shape=(None, None, 1),
            name='terminals',
        )

        if self._store_extra_policy_info:
            self._log_pis_ph = tf.placeholder(
                tf.float32,
                shape=(None, None, 1),
                name='log_pis',
            )
            self._raw_actions_ph = tf.placeholder(
                tf.float32,
                shape=(None, None, *self._action_shape),
                name='raw_actions',
            )

        # inner functions
        LOG_STD_MAX = 2
        LOG_STD_MIN = -20
        EPS = 1e-8

        def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None, kernel_initializer=None):
            for h in hidden_sizes[:-1]:
                x = tf.layers.dense(x, units=h, activation=activation, kernel_initializer=kernel_initializer)
            return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation,
                                   kernel_initializer=kernel_initializer)

        def gaussian_likelihood(x, mu, log_std):
            pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
            return tf.reduce_sum(pre_sum, axis=-1)

        def apply_squashing_func(mu, pi, logp_pi):
            # Adjustment to log prob
            # NOTE: This formula is a little bit magic. To get an understanding of where it
            # comes from, check out the original SAC paper (arXiv 1801.01290) and look in
            # appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi -= tf.reduce_sum(2 * (np.log(2) - pi - tf.nn.softplus(-2 * pi)), axis=-1)
            # Squash those unbounded actions!
            mu = tf.tanh(mu)
            pi = tf.tanh(pi)
            return mu, pi, logp_pi

        def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation):
            act_dim = a.shape.as_list()[-1]
            net = mlp(x, list(hidden_sizes), activation, activation)
            mu = tf.layers.dense(net, act_dim, activation=output_activation)
            log_std = tf.layers.dense(net, act_dim, activation=None)
            log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
            std = tf.exp(log_std)
            pi = mu + tf.random_normal(tf.shape(mu)) * std
            logp_pi = gaussian_likelihood(pi, mu, log_std)
            return mu, pi, logp_pi, std


        def mlp_actor_critic(x, x_v, a, hidden_sizes=(256, 256), activation=tf.nn.relu,
                             output_activation=None, policy=mlp_gaussian_policy):
            # policy
            with tf.variable_scope('pi'):
                mu, pi, logp_pi, std = policy(x, a, hidden_sizes, activation, output_activation)
                mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

            # vfs
            vf_mlp = lambda x: tf.squeeze(mlp(x, list(hidden_sizes) + [1], activation, None), axis=-1)

            with tf.variable_scope('q1'):
                q1 = vf_mlp(tf.concat([x_v, a], axis=-1))
            with tf.variable_scope('q2'):
                q2 = vf_mlp(tf.concat([x_v, a], axis=-1))
            return mu, pi, logp_pi, q1, q2, std

        policy_state1 = self._observations_ph
        value_state1 = self._observations_ph
        policy_state2 = value_state2 = self._next_observations_ph

        ac_kwargs = {
            "hidden_sizes": self.network_kwargs["hidden_sizes"],
            "activation": self.network_kwargs["activation"],
            "output_activation": self.network_kwargs["output_activation"]
        }

        with tf.variable_scope('main', reuse=False):
            self.mu, self.pi, logp_pi, q1, q2, std = mlp_actor_critic(policy_state1, value_state1, self._actions_ph,**ac_kwargs)


        pi_entropy = tf.reduce_sum(tf.log(std + 1e-8) + 0.5 * tf.log(2 * np.pi * np.e), axis=-1)
        with tf.variable_scope('main', reuse=True):
            # compose q with pi, for pi-learning
            _, _, _, q1_pi, q2_pi, _ = mlp_actor_critic(policy_state1, value_state1, self.pi, **ac_kwargs)
            # get actions and log probs of actions for next states, for Q-learning
            _, pi_next, logp_pi_next, _, _, _ = mlp_actor_critic(policy_state2, value_state2, self._actions_ph, **ac_kwargs)

        with tf.variable_scope('target'):
            # target q values, using actions from *current* policy
            _, _, _, q1_targ, q2_targ, _ = mlp_actor_critic(policy_state2, value_state2, pi_next, **ac_kwargs)

        # actions = self._policy.actions([self._observations_ph])
        # log_pis = self._policy.log_pis([self._observations_ph], actions)
        # assert log_pis.shape.as_list() == [None, 1]

        # alpha optimizer
        log_alpha = self._log_alpha = tf.get_variable(
            'log_alpha',
            dtype=tf.float32,
            initializer=0.0)
        alpha = tf.exp(log_alpha)
        if isinstance(self._target_entropy, Number):
            alpha_loss = -tf.reduce_mean(
                log_alpha * tf.stop_gradient(logp_pi + self._target_entropy))

            self._alpha_optimizer = tf.train.AdamOptimizer(
                self._policy_lr, name='alpha_optimizer')
            self._alpha_train_op = self._alpha_optimizer.minimize(
                loss=alpha_loss, var_list=[log_alpha])

            self._training_ops.update({
                'temperature_alpha': self._alpha_train_op
            })

        self._alpha = alpha
        assert self._action_prior == 'uniform'
        policy_prior_log_probs = 0.0
        # if self._action_prior == 'normal':
        #     policy_prior = tf.contrib.distributions.MultivariateNormalDiag(
        #         loc=tf.zeros(self._action_shape),
        #         scale_diag=tf.ones(self._action_shape))
        #     policy_prior_log_probs = policy_prior.log_prob(self.pi)
        # elif self._action_prior == 'uniform':
        #     policy_prior_log_probs = 0.0
        # else:
        #     raise NotImplementedError

        # Q_log_targets = tuple(
        #     Q([self._observations_ph, actions])
        #     for Q in self._Qs)
        # min_Q_log_target = tf.reduce_min(Q_log_targets, axis=0)

        min_q_pi = tf.minimum(q1_pi, q2_pi)
        min_q_targ = tf.minimum(q1_targ, q2_targ)

        if self._reparameterize:
            policy_kl_losses = (
                tf.stop_gradient(alpha) * logp_pi - min_q_pi - policy_prior_log_probs)
        else:
            raise NotImplementedError

        # assert policy_kl_losses.shape.as_list() == [None, 1]

        policy_loss = tf.reduce_mean(policy_kl_losses)

        # self._policy_optimizer = tf.train.AdamOptimizer(
        #     learning_rate=self._policy_lr,
        #     name="policy_optimizer")
        # policy_train_op = tf.contrib.layers.optimize_loss(
        #     policy_loss,
        #     self.global_step,
        #     learning_rate=self._policy_lr,
        #     optimizer=self._policy_optimizer,
        #     variables=self._policy.trainable_variables,
        #     increment_global_step=False,
        #     summaries=(
        #         "loss", "gradients", "gradient_norm", "global_gradient_norm"
        #     ) if self._tf_summaries else ())
        # self._training_ops.update({'policy_train_op': policy_train_op})


        # Q
        # next_actions = self._policy.actions([self._next_observations_ph])
        next_log_pis = logp_pi_next
        min_next_Q = min_q_targ
        next_value = min_next_Q - self._alpha * next_log_pis

        q_target = td_target(
            reward=self._reward_scale * self._rewards_ph,
            discount=self._discount, next_value=(1 - self._terminals_ph) * next_value)

        # assert q_target.shape.as_list() == [None, 1]
        # (self._Q_values,
        #  self._Q_losses,
        #  self._alpha,
        #  self.global_step),
        self.Q = q1 + q2

        q1_loss = tf.reduce_mean(tf.square(q_target - q1))
        q2_loss = tf.reduce_mean(tf.square(q_target - q2))
        value_loss = q1_loss + q2_loss
        self.Q_loss = (tf.square(q_target - q1) + tf.square(q_target - q2)) / 2

        value_optimizer = tf.train.AdamOptimizer(learning_rate=self._Q_lr)
        value_params = get_vars('main/q')
        if self.adapt:
            value_params += get_vars("lstm_net_v")

        grads, variables = zip(*value_optimizer.compute_gradients(value_loss, var_list=value_params))

        _, q_global_norm = tf.clip_by_global_norm(grads, 2000)
        train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

        pi_optimizer = tf.train.AdamOptimizer(learning_rate=self._policy_lr)
        pi_var_list = get_vars('main/pi')
        if self.adapt:
            pi_var_list += get_vars("lstm_net_pi")
        train_pi_op = pi_optimizer.minimize(policy_loss, var_list=pi_var_list)
        _, pi_global_norm = tf.clip_by_global_norm(grads, 2000)
        alpha_loss = - alpha * tf.stop_gradient(
            tf.reduce_mean(self._target_entropy + logp_pi))
        alpha_optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
        if self.optim_alpha:
            train_alpha_op = alpha_optimizer.minimize(alpha_loss, var_list=[log_alpha])
        else:
            train_alpha_op = tf.no_op()


        with tf.control_dependencies([train_value_op]):
            target_update = tf.group([tf.assign(v_targ, self._tau * v_targ + (1 - self._tau) * v_main)
                                      for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        self.target_init = tf.group([tf.assign(v_targ, v_main)
                                     for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        # construct opt
        self._training_ops = [tf.group((train_value_op, train_pi_op, target_update, train_alpha_op)),
                              { "sac_pi/pi_global_norm": pi_global_norm,
                                "sac_Q/q_global_norm": q_global_norm,
                                "Q/q1_loss": q1_loss,
                                "sac_Q/q2_loss": q2_loss,
                                "sac_Q/q1": q1,
                                "sac_Q/q2": q2,
                                "sac_pi/alpha": alpha,
                                "sac_pi/pi_entropy": pi_entropy,
                                "sac_pi/logp_pi": logp_pi,
                                "sac_pi/std": logp_pi, }]

        self._session.run(tf.global_variables_initializer())

    def get_action_meta(self, state, hidden, deterministic=False):
        with self._session.as_default():
            state_dim = len(np.shape(state))
            if state_dim == 2:
                state = state[None]
            feed_dict = {
                self._observations_ph: state,
                self._prev_state_p_ph: hidden
            }
            mu, pi = self._session.run([self.mu, self.pi], feed_dict=feed_dict)
            if state_dim == 2:
                mu = mu[0]
                pi = pi[0]
            # print(f"[ DEBUG ]: pi_shape: {pi.shape}, mu_shape: {mu.shape}")
            if deterministic:
                return mu, hidden
            else:
                return pi, hidden

    def make_init_hidden(self, batch_size=1):
        return np.zeros((batch_size, self.gru_state_dim))

    def _train(self):
        
        """Return a generator that performs RL training.

        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy (`Policy`): Policy used for training
            initial_exploration_policy ('Policy'): Policy used for exploration
                If None, then all exploration is done using policy
            pool (`PoolBase`): Sample pool to add samples to
        """
        training_environment = self._training_environment
        evaluation_environment = self._evaluation_environment
        # policy = self._policy
        pool = self._pool
        model_metrics = {}

        # if not self._training_started:
        self._init_training()
        # TODO: change policy to placeholder or a function
        def get_action(state, hidden, deterministic=False):
            return self.get_action_meta(state, hidden, deterministic)
        def make_init_hidden(batch_size=1):
            return self.make_init_hidden(batch_size)
        self.sampler.initialize(training_environment, (get_action, make_init_hidden), pool)

        gt.reset_root()
        gt.rename_root('RLAlgorithm')
        gt.set_def_unique(False)

        # self._training_before_hook()

        #### model training
        print('[ MOPO ] log_dir: {} | ratio: {}'.format(self._log_dir, self._real_ratio))
        print('[ MOPO ] Training model at epoch {} | freq {} | timestep {} (total: {})'.format(
            self._epoch, self._model_train_freq, self._timestep, self._total_timestep)
        )
        # train dynamics model offline
        max_epochs = 1 if self._model.model_loaded else None
        model_train_metrics = self._train_model(batch_size=256, max_epochs=max_epochs, holdout_ratio=0.2, max_t=self._max_model_t)

        model_metrics.update(model_train_metrics)
        self._log_model()
        gt.stamp('epoch_train_model')
        #### 
        tester.time_step_holder.set_time(0)
        for self._epoch in gt.timed_for(range(self._epoch, self._n_epochs)):

            self._epoch_before_hook()
            gt.stamp('epoch_before_hook')

            self._training_progress = Progress(self._epoch_length * self._n_train_repeat)
            start_samples = self.sampler._total_samples
            training_logs = {}
            for timestep in count():
                self._timestep = timestep
                if (timestep >= self._epoch_length
                    and self.ready_to_train):
                    break

                self._timestep_before_hook()
                gt.stamp('timestep_before_hook')

                ## model rollouts
                if timestep % self._model_train_freq == 0 and self._real_ratio < 1.0:
                    self._training_progress.pause()
                    self._set_rollout_length()
                    self._reallocate_model_pool()
                    model_rollout_metrics = self._rollout_model(rollout_batch_size=self._rollout_batch_size,
                                                                deterministic=self._deterministic)
                    model_metrics.update(model_rollout_metrics)
                    
                    gt.stamp('epoch_rollout_model')
                    self._training_progress.resume()

                ## train actor and critic
                if self.ready_to_train:
                    # print('[ DEBUG ]: ready to train at timestep: {}'.format(timestep))
                    training_logs = self._do_training_repeats(timestep=timestep)
                gt.stamp('train')

                self._timestep_after_hook()
                gt.stamp('timestep_after_hook')

            training_paths = self.sampler.get_last_n_paths(
                math.ceil(self._epoch_length / self.sampler._max_path_length))
            # evaluate the polices
            evaluation_paths = self._evaluation_paths(
                (lambda _state, _hidden: get_action(_state, _hidden, True), make_init_hidden), evaluation_environment)
            gt.stamp('evaluation_paths')
            if evaluation_paths:
                evaluation_metrics = self._evaluate_rollouts(
                    evaluation_paths, evaluation_environment)
                gt.stamp('evaluation_metrics')
            else:
                evaluation_metrics = {}

            gt.stamp('epoch_after_hook')

            sampler_diagnostics = self.sampler.get_diagnostics()

            diagnostics = self.get_diagnostics(
                iteration=self._total_timestep,
                batch=self._evaluation_batch(),
                training_paths=training_paths,
                evaluation_paths=evaluation_paths)

            time_diagnostics = gt.get_times().stamps.itrs

            diagnostics.update(OrderedDict((
                *(
                    ('evaluation/{}'.format(key), evaluation_metrics[key])
                    for key in sorted(evaluation_metrics.keys())
                ),
                *(
                    ('times/{}'.format(key), time_diagnostics[key][-1])
                    for key in sorted(time_diagnostics.keys())
                ),
                *(
                    ('sampler/{}'.format(key), sampler_diagnostics[key])
                    for key in sorted(sampler_diagnostics.keys())
                ),
                *(
                    ('model/{}'.format(key), model_metrics[key])
                    for key in sorted(model_metrics.keys())
                ),
                ('epoch', self._epoch),
                ('timestep', self._timestep),
                ('timesteps_total', self._total_timestep),
                ('train-steps', self._num_train_steps),
                *(
                    ('training/{}'.format(key), training_logs[key])
                    for key in sorted(training_logs.keys())
                )
            )))
            diagnostics['perf/AverageReturn'] = diagnostics['evaluation/return-average']
            diagnostics['perf/AverageLength'] = diagnostics['evaluation/episode-length-avg']
            if not self.min_ret == self.max_ret:
                diagnostics['perf/NormalizedReturn'] = (diagnostics['perf/AverageReturn'] - self.min_ret) \
                                                       / (self.max_ret - self.min_ret)
            # diagnostics['keys/logp_pi'] =  diagnostics['training/sac_pi/logp_pi']
            if self._eval_render_mode is not None and hasattr(
                    evaluation_environment, 'render_rollouts'):
                training_environment.render_rollouts(evaluation_paths)

            ## ensure we did not collect any more data
            assert self._pool.size == self._init_pool_size
            for k, v in diagnostics.items():
                # print('[ DEBUG ] epoch: {} diagnostics k: {}, v: {}'.format(self._epoch, k, v))
                self._writer.add_scalar(k, v, self._epoch)
            yield diagnostics

        self.sampler.terminate()

        self._training_after_hook()

        self._training_progress.close()

        yield {'done': True, **diagnostics}

    def train(self, *args, **kwargs):
        return self._train(*args, **kwargs)

    def _log_policy(self):
        # TODO: how to saving models
        save_path = os.path.join(self._log_dir, 'models')
        filesystem.mkdir(save_path)
        weights = self._policy.get_weights()
        data = {'policy_weights': weights}
        full_path = os.path.join(save_path, 'policy_{}.pkl'.format(self._total_timestep))
        print('Saving policy to: {}'.format(full_path))
        pickle.dump(data, open(full_path, 'wb'))

    def _log_model(self):
        print('[ MODEL ]: {}'.format(self._model_type))
        if self._model_type == 'identity':
            print('[ MOPO ] Identity model, skipping save')
        elif self._model.model_loaded:
            print('[ MOPO ] Loaded model, skipping save')
        else:
            save_path = os.path.join(self._log_dir, 'models')
            filesystem.mkdir(save_path)
            print('[ MOPO ] Saving model to: {}'.format(save_path))
            self._model.save(save_path, self._total_timestep)

    def _set_rollout_length(self):
        min_epoch, max_epoch, min_length, max_length = self._rollout_schedule
        if self._epoch <= min_epoch:
            y = min_length
        else:
            dx = (self._epoch - min_epoch) / (max_epoch - min_epoch)
            dx = min(dx, 1)
            y = dx * (max_length - min_length) + min_length

        self._rollout_length = int(y)
        print('[ Model Length ] Epoch: {} (min: {}, max: {}) | Length: {} (min: {} , max: {})'.format(
            self._epoch, min_epoch, max_epoch, self._rollout_length, min_length, max_length
        ))

    def _reallocate_model_pool(self):
        obs_space = self._pool._observation_space
        act_space = self._pool._action_space

        rollouts_per_epoch = self._rollout_batch_size * self._epoch_length / self._model_train_freq
        model_steps_per_epoch = int(self._rollout_length * rollouts_per_epoch)
        new_pool_size = self._model_retain_epochs * model_steps_per_epoch

        if not hasattr(self, '_model_pool'):
            print('[ MOPO ] Initializing new model pool with size {:.2e}'.format(
                new_pool_size
            ))
            self._model_pool = SimpleReplayPool(obs_space, act_space, new_pool_size)
        
        elif self._model_pool._max_size != new_pool_size:
            print('[ MOPO ] Updating model pool | {:.2e} --> {:.2e}'.format(
                self._model_pool._max_size, new_pool_size
            ))
            samples = self._model_pool.return_all_samples()
            new_pool = SimpleReplayPool(obs_space, act_space, new_pool_size)
            new_pool.add_samples(samples)
            assert self._model_pool.size == new_pool.size
            self._model_pool = new_pool

    def _train_model(self, **kwargs):
        if self._model_type == 'identity':
            print('[ MOPO ] Identity model, skipping model')
            model_metrics = {}
        else:
            env_samples = self._pool.return_all_samples()
            train_inputs, train_outputs = format_samples_for_training(env_samples)
            model_metrics = self._model.train(train_inputs, train_outputs, **kwargs)
        return model_metrics

    def _rollout_model(self, rollout_batch_size, **kwargs):
        print('[ Model Rollout ] Starting | Epoch: {} | Rollout length: {} | Batch size: {} | Type: {}'.format(
            self._epoch, self._rollout_length, rollout_batch_size, self._model_type
        ))
        batch = self.sampler.random_batch(rollout_batch_size)
        obs = batch['observations']
        steps_added = []
        for i in range(self._rollout_length):
            hidden = self.make_init_hidden(1)
            if not self._rollout_random:
                # act = self._policy.actions_np(obs)
                act, hidden = self.get_action_meta(obs, hidden)
            else:
                # act_ = self._policy.actions_np(obs)
                act_, hidden = self.get_action_meta(obs, hidden)
                act = np.random.uniform(low=-1, high=1, size=act_.shape)

            if self._model_type == 'identity':
                next_obs = obs
                rew = np.zeros((len(obs), 1))
                term = (np.ones((len(obs), 1)) * self._identity_terminal).astype(np.bool)
                info = {}
            else:
                next_obs, rew, term, info = self.fake_env.step(obs, act, **kwargs)
            steps_added.append(len(obs))

            samples = {'observations': obs, 'actions': act, 'next_observations': next_obs, 'rewards': rew, 'terminals': term}
            self._model_pool.add_samples(samples)

            nonterm_mask = ~term.squeeze(-1)
            if nonterm_mask.sum() == 0:
                print('[ Model Rollout ] Breaking early: {} | {} / {}'.format(i, nonterm_mask.sum(), nonterm_mask.shape))
                break

            obs = next_obs[nonterm_mask]

        mean_rollout_length = sum(steps_added) / rollout_batch_size
        rollout_stats = {'mean_rollout_length': mean_rollout_length}
        print('[ Model Rollout ] Added: {:.1e} | Model pool: {:.1e} (max {:.1e}) | Length: {} | Train rep: {}'.format(
            sum(steps_added), self._model_pool.size, self._model_pool._max_size, mean_rollout_length, self._n_train_repeat
        ))
        return rollout_stats

    def _visualize_model(self, env, timestep):
        ## save env state
        state = env.unwrapped.state_vector()
        qpos_dim = len(env.unwrapped.sim.data.qpos)
        qpos = state[:qpos_dim]
        qvel = state[qpos_dim:]

        print('[ Visualization ] Starting | Epoch {} | Log dir: {}\n'.format(self._epoch, self._log_dir))
        visualize_policy(env, self.fake_env, self._policy, self._writer, timestep)
        print('[ Visualization ] Done')
        ## set env state
        env.unwrapped.set_state(qpos, qvel)

    def _do_training_repeats(self, timestep):
        """Repeat training _n_train_repeat times every _train_every_n_steps"""
        if timestep % self._train_every_n_steps > 0: return
        trained_enough = (
                self._train_steps_this_epoch
                > self._max_train_repeat_per_timestep * self._timestep)
        if trained_enough: return
        log_buffer = []
        logs = {}
        for i in range(self._n_train_repeat):
            logs = self._do_training(
                iteration=timestep,
                batch=self._training_batch())
            log_buffer.append(logs)
        logs_buffer = {k: np.mean([item[k] for item in log_buffer]) for k in logs}

        self._num_train_steps += self._n_train_repeat
        self._train_steps_this_epoch += self._n_train_repeat
        return logs_buffer

    def _training_batch(self, batch_size=None):
        batch_size = batch_size or self.sampler._batch_size
        env_batch_size = int(batch_size*self._real_ratio)
        model_batch_size = batch_size - env_batch_size
        # TODO: how to set teriminal state.
        # TODO: how to set model pool.

        ## can sample from the env pool even if env_batch_size == 0
        env_batch = self._pool.random_batch(env_batch_size)

        if model_batch_size > 0:
            model_batch = self._model_pool.random_batch(model_batch_size)

            # keys = env_batch.keys()
            keys = set(env_batch.keys()) & set(model_batch.keys())
            batch = {k: np.concatenate((env_batch[k], model_batch[k]), axis=0) for k in keys}
        else:
            ## if real_ratio == 1.0, no model pool was ever allocated,
            ## so skip the model pool sampling
            batch = env_batch
        return batch

    # def _init_global_step(self):
    #     self.global_step = training_util.get_or_create_global_step()
    #     self._training_ops.update({
    #         'increment_global_step': training_util._increment_global_step(1)
    #     })
    #

    def _init_training(self):
        self._session.run(self.target_init)
        # self._update_target(tau=1.0)

    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""

        self._training_progress.update()
        self._training_progress.set_description()

        feed_dict = self._get_feed_dict(iteration, batch)

        res = self._session.run(self._training_ops, feed_dict)
        logs = {k: np.mean(res[1][k]) for k in res[1]}
        # for k, v in logs.items():
        #     print("[ DEBUG ] k: {}, v: {}".format(k, v))
        #     self._writer.add_scalar(k, v, iteration)
        return logs
        # if iteration % self._target_update_interval == 0:
        #     # Run target ops here.
        #     self._update_target()

    def _get_feed_dict(self, iteration, batch):
        """Construct TensorFlow feed_dict from sample batch."""
        state_dim = len(batch['observations'].shape)
        resize = lambda x: x[None] if state_dim == 2 else x
        feed_dict = {
            self._observations_ph: resize(batch['observations']),
            self._actions_ph: resize(batch['actions']),
            self._next_observations_ph: resize(batch['next_observations']),
            self._rewards_ph: resize(batch['rewards']),
            self._terminals_ph: resize(batch['terminals']),
        }

        if self._store_extra_policy_info:
            feed_dict[self._log_pis_ph] = resize(batch['log_pis'])
            feed_dict[self._raw_actions_ph] = resize(batch['raw_actions'])

        if iteration is not None:
            feed_dict[self._iteration_ph] = iteration

        return feed_dict

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        """Return diagnostic information as ordered dictionary.

        Records mean and standard deviation of Q-function and state
        value function, and TD-loss (mean squared Bellman error)
        for the sample batch.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        feed_dict = self._get_feed_dict(iteration, batch)

        (Q_values, Q_losses, alpha, global_step) = self._session.run(
            (self.Q,
             self.Q_loss,
             self._alpha,
             self.global_step),
            feed_dict)

        diagnostics = OrderedDict({
            'Q-avg': np.mean(Q_values),
            'Q-std': np.std(Q_values),
            'Q_loss': np.mean(Q_losses),
            'alpha': alpha,
        })

        # TODO (luofm): policy diagnostics
        # policy_diagnostics = self._policy.get_diagnostics(
        #     batch['observations'])
        # diagnostics.update({
        #     'policy/{}'.format(key): value
        #     for key, value in policy_diagnostics.items()
        # })

        if self._plotter:
            self._plotter.draw()

        return diagnostics

    @property
    def tf_saveables(self):
        saveables = {
            '_policy_optimizer': self._policy_optimizer,
            **{
                'Q_optimizer_{}'.format(i): optimizer
                for i, optimizer in enumerate(self._Q_optimizers)
            },
            '_log_alpha': self._log_alpha,
        }

        if hasattr(self, '_alpha_optimizer'):
            saveables['_alpha_optimizer'] = self._alpha_optimizer

        return saveables
