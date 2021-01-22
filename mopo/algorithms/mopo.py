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
from softlearning.replay_pools.simple_replay_pool import SimpleReplayPool, SimpleReplayTrajPool

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
            tester=None,
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
        self.tester = tester
        if '_smv' in model_name:
            self._env_name = model_name[:-8] + '-v0'
        else:
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
        self.fix_rollout_length = rollout_length
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
        self.gru_state_dim = network_kwargs['lstm_hidden_unit']
        self.network_kwargs = network_kwargs
        self.adapt = adapt
        self.optim_alpha = False
        # self._policy = policy

        # self._Qs = Qs
        # self._Q_targets = tuple(tf.keras.models.clone_model(Q) for Q in Qs)

        observation_shape = self._training_environment.active_observation_shape
        action_shape = self._training_environment.action_space.shape
        self._pool = pool
        print('[ DEBUG ] pool.size=', pool.size)

        if self.adapt:
            self._env_pool = SimpleReplayTrajPool(
                training_environment.observation_space, training_environment.action_space, self.fix_rollout_length, 2e5
            )
        else:
            self._env_pool = self._pool

        self._plotter = plotter
        self._tf_summaries = tf_summaries

        self._policy_lr = lr
        self._Q_lr = lr

        self._reward_scale = reward_scale
        self._target_entropy = (
            -np.prod(self._training_environment.action_space.shape)
            if target_entropy == 'auto'
            else target_entropy)
        print('[ MOPO ] Target entropy: {}'.format(self._target_entropy))

        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval
        self._action_prior = action_prior

        self._reparameterize = reparameterize
        self._store_extra_policy_info = store_extra_policy_info


        assert len(observation_shape) == 1, observation_shape
        self._observation_shape = observation_shape
        assert len(action_shape) == 1, action_shape
        self._action_shape = action_shape

        self._build()

        #### load replay pool data
        self._pool_load_path = pool_load_path
        self._pool_load_max_size = pool_load_max_size

        loader.restore_pool(self._pool, self._pool_load_path, self._pool_load_max_size, save_path=self._log_dir, adapt=False, maxlen=self.fix_rollout_length)
        if self.adapt:
            loader.restore_pool(self._env_pool, self._pool_load_path, self._pool_load_max_size, save_path=self._log_dir, adapt=self.adapt, maxlen=self.fix_rollout_length)
        print('[ DEBUG ] pool.size (after restore from pool) =', pool.size)
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

        self._last_actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, None, *self._action_shape),
            name='lst_actions',
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

        self._valid_ph = tf.placeholder(
            tf.float32,
            shape=(None, None, 1),
            name='valid',
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
        valid_num = tf.reduce_sum(tf.reduce_sum(self._valid_ph[:, :, 0]))
        def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None, kernel_initializer=None):
            print('[ DEBUG ], hidden layer size: ', hidden_sizes)
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
            print('[ DEBUG ]: output activation: ', output_activation, ', activation: ', activation)
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

        def lstm_emb(x_ph, a_ph, pre_state_p, pre_state_v, seq_len=self.seq_len):
            state_acs = tf.concat([x_ph, a_ph], axis=-1)

            with tf.variable_scope("lstm_net_p", reuse=tf.AUTO_REUSE):
                lstm_input = state_acs

                # cells_policy = []
                # for h in self.lstm_hidden_layers:
                #     cells_policy.append(tf.nn.rnn_cell.GRUCell(num_units=h))
                # cells_policy = tf.nn.rnn_cell.MultiRNNCell(cells=cells_policy,
                #                                                 state_is_tuple=False)
                cells_policy = tf.nn.rnn_cell.GRUCell(num_units=self.network_kwargs["lstm_hidden_unit"])
                policy_out, next_policy_hidden_out = tf.nn.dynamic_rnn(cells_policy, lstm_input,
                                                                       initial_state=pre_state_p,
                                                                       dtype=tf.float32, sequence_length=seq_len)
                policy_out = mlp(policy_out, hidden_sizes=[self.network_kwargs['embedding_size']],
                                 activation=tf.tanh, output_activation=tf.tanh)
                policy_state = tf.concat([policy_out, x_ph], axis=-1)

            with tf.variable_scope("lstm_net_v", reuse=tf.AUTO_REUSE):
                # cells_policy = []
                # for h in self.lstm_hidden_layers:
                #     cells_policy.append(tf.nn.rnn_cell.GRUCell(num_units=h))
                # cells_policy = tf.nn.rnn_cell.MultiRNNCell(cells=cells_policy,
                #                                                 state_is_tuple=False)
                lstm_input = state_acs
                cells_value = tf.nn.rnn_cell.GRUCell(num_units=self.network_kwargs["lstm_hidden_unit"])
                value_out, next_value_hidden_out = tf.nn.dynamic_rnn(cells_value, lstm_input,
                                                                     initial_state=pre_state_v,
                                                                     dtype=tf.float32, sequence_length=seq_len)
                value_out = mlp(value_out, hidden_sizes=[self.network_kwargs['embedding_size']],
                                 activation=tf.tanh, output_activation=tf.tanh)
                value_state = tf.concat([value_out, x_ph], axis=-1)
            return policy_state, value_state, next_policy_hidden_out, next_value_hidden_out, policy_out, value_out

        if self.adapt:
            # input_x = tf.concat((self.x_ph, self._next_observations_ph[:, -1:, :]), axis=1)
            # input_lst_a = tf.concat((self._last_actions_ph, self._actions_ph[:, -1:, :]), axis=1)
            policy_state, value_state, self.next_policy_hidden_out, self.next_value_hidden_out, policy_out, value_out = lstm_emb(
                self._observations_ph, self._last_actions_ph, self._prev_state_p_ph, self._prev_state_v_ph)
            policy_state_next, value_state_next, _, _, policy_out, value_out = lstm_emb(
                self._next_observations_ph[:, -1:], self._actions_ph[:, -1:], self.next_policy_hidden_out,
                self.next_value_hidden_out, tf.ones_like(self.seq_len))
            policy_state1, value_state1 = policy_state, value_state
            policy_state2, value_state2 = tf.concat((policy_state[:, 1:], policy_state_next), axis=1), \
                                          tf.concat((value_state[:, 1:], value_state_next), axis=1)
            action_ph = self._actions_ph
            reward_ph = self._rewards_ph
            done_ph = self._terminals_ph
        else:
            policy_state1 = self._observations_ph
            value_state1 = self._observations_ph
            policy_state2 = value_state2 = self._next_observations_ph
            action_ph = self._actions_ph
            reward_ph = self._rewards_ph
            done_ph = self._terminals_ph
            self.next_value_hidden_out = self._prev_state_v_ph
            self.next_policy_hidden_out = self._prev_state_p_ph

        ac_kwargs = {
            "hidden_sizes": self.network_kwargs["hidden_sizes"],
            "activation": self.network_kwargs["activation"],
            "output_activation": self.network_kwargs["output_activation"]
        }

        with tf.variable_scope('main', reuse=False):
            self.mu, self.pi, logp_pi, q1, q2, std = mlp_actor_critic(policy_state1, value_state1, action_ph, **ac_kwargs)


        pi_entropy = tf.reduce_sum(tf.log(std + 1e-8) + 0.5 * tf.log(2 * np.pi * np.e), axis=-1)
        with tf.variable_scope('main', reuse=True):
            # compose q with pi, for pi-learning
            _, _, _, q1_pi, q2_pi, _ = mlp_actor_critic(policy_state1, value_state1, self.pi, **ac_kwargs)
            # get actions and log probs of actions for next states, for Q-learning
            _, pi_next, logp_pi_next, _, _, _ = mlp_actor_critic(policy_state2, value_state2, action_ph, **ac_kwargs)

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

        self._alpha = alpha
        assert self._action_prior == 'uniform'
        policy_prior_log_probs = 0.0

        min_q_pi = tf.minimum(q1_pi, q2_pi)
        min_q_targ = tf.minimum(q1_targ, q2_targ)

        if self._reparameterize:
            policy_kl_losses = (
                tf.stop_gradient(alpha) * logp_pi - min_q_pi - policy_prior_log_probs)
        else:
            raise NotImplementedError


        policy_loss = tf.reduce_sum(policy_kl_losses * self._valid_ph[:, :, 0]) / valid_num

        # Q
        next_log_pis = logp_pi_next
        min_next_Q = min_q_targ
        next_value = min_next_Q - self._alpha * next_log_pis

        q_target = td_target(
            reward=self._reward_scale * reward_ph[..., 0],
            discount=self._discount, next_value=(1 - done_ph[..., 0]) * next_value)



        # assert q_target.shape.as_list() == [None, 1]
        # (self._Q_values,
        #  self._Q_losses,
        #  self._alpha,
        #  self.global_step),
        self.Q1 = q1
        self.Q2 = q2
        q_target = tf.stop_gradient(q_target)
        q1_loss = tf.reduce_sum(tf.square((q_target - q1) * self._valid_ph[:, :, 0])) * 0.5 / valid_num
        q2_loss = tf.reduce_sum(tf.square((q_target - q2) * self._valid_ph[:, :, 0])) * 0.5 / valid_num

        # q1_loss = tf.losses.mean_squared_error(labels=q_target, predictions=q1, weights=0.5)
        # q2_loss = tf.losses.mean_squared_error(labels=q_target, predictions=q2, weights=0.5)
        self.Q_loss = (q1_loss + q2_loss) / 2

        value_optimizer1 = tf.train.AdamOptimizer(learning_rate=self._Q_lr)
        value_optimizer2 = tf.train.AdamOptimizer(learning_rate=self._Q_lr)
        print('q1_pi: {}, q2_pi: {}, policy_state2: {}, policy_state1: {}, '
              'tmux a: {}, q_targ: {}, mu: {}, reward: {}, '
              'terminal: {}, target_q: {}, next_value: {}, '
              'q1: {}, logp_pi: {}, min_q_pi: {}, q1_loss: {}, next_hidden_out: {}'.format(q1_pi, q2_pi, policy_state1, policy_state2, pi_next,
                                                         q1_targ, self.mu, self._rewards_ph[..., 0],
                                                         self._terminals_ph[..., 0],
                                                         q_target, next_value, q1, logp_pi, min_q_pi, q1_loss, self.next_policy_hidden_out))
        print('[ DEBUG ]: Q lr is {}'.format(self._Q_lr))


        # train_value_op = value_optimizer.apply_gradients(zip(grads, variables))
        pi_optimizer = tf.train.AdamOptimizer(learning_rate=self._policy_lr)
        print('[ DEBUG ]: policy lr is {}'.format(self._policy_lr))

        pi_var_list = get_vars('main/pi')
        if self.adapt:
            pi_var_list += get_vars("lstm_net_pi")
        train_pi_op = pi_optimizer.minimize(policy_loss, var_list=pi_var_list)
        pgrads, variables = zip(*pi_optimizer.compute_gradients(policy_loss, var_list=pi_var_list))

        _, pi_global_norm = tf.clip_by_global_norm(pgrads, 2000)

        # TODO (luofm): figure out why ``core dumped'' after add the following line
        # with tf.control_dependencies([train_pi_op]):
        value_params1 = get_vars('main/q1')
        value_params2 = get_vars('main/q2')
        if self.adapt:
            value_params1 += get_vars("lstm_net_v")
            value_params2 += get_vars("lstm_net_v")

        grads, variables = zip(*value_optimizer1.compute_gradients(q1_loss, var_list=value_params1))
        _, q_global_norm = tf.clip_by_global_norm(grads, 2000)
        train_value_op1 = value_optimizer1.minimize(q1_loss, var_list=value_params1)
        train_value_op2 = value_optimizer2.minimize(q2_loss, var_list=value_params2)
        # with tf.control_dependencies([train_value_op1, train_value_op2]):
        if isinstance(self._target_entropy, Number):
            alpha_loss = -tf.reduce_sum((
                log_alpha * tf.stop_gradient(logp_pi + self._target_entropy)) * self._valid_ph[:, :, 0]) / valid_num
            self._alpha_optimizer = tf.train.AdamOptimizer(self._policy_lr, name='alpha_optimizer')
            self._alpha_train_op = self._alpha_optimizer.minimize(
                loss=alpha_loss, var_list=[log_alpha])
        else:
            self._alpha_train_op = tf.no_op()


        self.target_update = tf.group([tf.assign(v_targ, (1 - self._tau) * v_targ + self._tau * v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        self.target_init = tf.group([tf.assign(v_targ, v_main)
                                     for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        # construct opt
        self._training_ops = [
            tf.group((train_value_op2, train_value_op1, train_pi_op, self._alpha_train_op)),
                              { "sac_pi/pi_global_norm": pi_global_norm,
                                # "sac_Q/q_global_norm": q_global_norm,
                                "Q/q1_loss": q1_loss,
                                "sac_Q/q2_loss": q2_loss,
                                "sac_Q/q1": q1,
                                "sac_Q/q2": q2,
                                "sac_pi/alpha": alpha,
                                "sac_pi/pi_entropy": pi_entropy,
                                "sac_pi/logp_pi": logp_pi,
                                "sac_pi/std": logp_pi,
                                "sac_pi/valid_num": valid_num,
                                "sac_pi/policy_loss": policy_loss,
                                "sac_pi/alpha_loss": alpha_loss
                              }]
        # self._training_ops = [q1_loss, q2_loss, policy_loss, alpha_loss] #, logp_pi]

        self._session.run(tf.global_variables_initializer())
        self._session.run(self.target_init)

    def get_action_meta(self, state, hidden, deterministic=False):
        assert isinstance(hidden, tuple), 'hidden: (hidden state of policy, lst_action)'
        with self._session.as_default():
            state_dim = len(np.shape(state))
            if state_dim == 2:
                state = np.expand_dims(state, 1)
            feed_dict = {
                self._observations_ph: state,
                self._prev_state_p_ph: hidden[0],
                self._last_actions_ph: hidden[1],
                self.seq_len:[1] * state.shape[0]

            }
            mu, pi, next_hidden = self._session.run([self.mu, self.pi, self.next_policy_hidden_out], feed_dict=feed_dict)
            mu_origin, pi_origin = mu, pi
            if state_dim == 2:
                mu = mu[:, 0]
                pi = pi[:, 0]

            # print(f"[ DEBUG ]: pi_shape: {pi.shape}, mu_shape: {mu.shape}")
            if deterministic:
                hidden = (next_hidden, mu_origin)
                return mu, hidden
            else:
                hidden = (next_hidden, pi_origin)
                return pi, hidden

    def make_init_hidden(self, batch_size=1):
        res = (np.zeros((batch_size, self.gru_state_dim)), np.zeros((batch_size, 1, self._action_shape[0])))
        return res

    def mask_hidden(self, hidden, mask):
        res = (hidden[0][mask], hidden[1][mask])
        return res

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
                                                                deterministic=self._deterministic,
                                                                mask_hidden=self.mask_hidden)
                    model_metrics.update(model_rollout_metrics)
                    # print('[ DEBUG ] after update of model metrics')
                    gt.stamp('epoch_rollout_model')
                    self._training_progress.resume()
                    # print('[ DEBUG ] after resume')
                # print('[ DEBUG ]: judge ready to train... {}'.format(self.ready_to_train))
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
            if self._epoch % 20 == 0:
                self.tester.sync_log_file()
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
            if self.adapt:
                self._model_pool = SimpleReplayTrajPool(obs_space, act_space, self.fix_rollout_length, new_pool_size)
            else:
                self._model_pool = SimpleReplayPool(obs_space, act_space, new_pool_size)

        elif self._model_pool._max_size != new_pool_size:
            print('[ MOPO ] Updating model pool | {:.2e} --> {:.2e}'.format(
                self._model_pool._max_size, new_pool_size
            ))
            samples = self._model_pool.return_all_samples()
            # new_pool = SimpleReplayPool(obs_space, act_space, new_pool_size)
            if self.adapt:
                new_pool = SimpleReplayTrajPool(obs_space, act_space, self.fix_rollout_length, new_pool_size)
            else:
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

    def _rollout_model(self, rollout_batch_size, mask_hidden, **kwargs):
        print('[ Model Rollout ] Starting | Epoch: {} | Rollout length: {} | Batch size: {} | Type: {} | Adapt: {}'.format(
            self._epoch, self._rollout_length, rollout_batch_size, self._model_type, self.adapt
        ))
        batch = self.sampler.random_batch(rollout_batch_size)
        obs = batch['observations']
        steps_added = []
        hidden = self.make_init_hidden(obs.shape[0])
        current_nonterm = np.ones((len(obs)), dtype=bool)
        sample_list = []
        for i in range(self._rollout_length):
            # print('[ DEBUG ] obs shape: {}'.format(obs.shape))
            lst_action = hidden[1]
            if not self._rollout_random:
                # act = self._policy.actions_np(obs)
                act, hidden = self.get_action_meta(obs, hidden)
            else:
                # act_ = self._policy.actions_np(obs)
                act_, hidden = self.get_action_meta(obs, hidden)
                act = np.random.uniform(low=-1, high=1, size=act_.shape)
            print('[ DEBUG ] obs shape: {}, act shape: {}, non term number: {}'.format(obs.shape, act.shape, current_nonterm.sum()))
            if self._model_type == 'identity':
                next_obs = obs
                rew = np.zeros((len(obs), 1))
                term = (np.ones((len(obs), 1)) * self._identity_terminal).astype(np.bool)
                info = {}
            else:
                # print("act: {}, obs: {}".format(act.shape, obs.shape))
                next_obs, rew, term, info = self.fake_env.step(obs, act, **kwargs)
            steps_added.append(len(obs))
            nonterm_mask = ~term.squeeze(-1)
            assert current_nonterm.shape == nonterm_mask.shape
            # print('size of last action: ', lst_action.shape, obs.shape, lst_action.squeeze(1).shape)
            samples = {'observations': obs, 'actions': act, 'next_observations': next_obs,
                       'rewards': rew, 'terminals': term,
                       'last_actions': lst_action.squeeze(1), 'valid': current_nonterm.reshape(-1, 1)}
            if not self.adapt:
                self._model_pool.add_samples(samples)
            else:
                samples = {k: np.expand_dims(v, 1) for k, v in samples.items()}
                sample_list.append(samples)
            current_nonterm = current_nonterm & nonterm_mask
            obs = next_obs
            # if nonterm_mask.sum() == 0:
            #     print(
            #         '[ Model Rollout ] Breaking early: {} | {} / {}'.format(i, nonterm_mask.sum(), nonterm_mask.shape))
            #     break
        if self.adapt:
            samples = {}
            for k in sample_list[0]:
                data = np.concatenate([item[k] for item in sample_list], axis=1)
                samples[k] = data
                # print('[ DEBUG ]: shape of data: ', np.shape(data))
                # print(k, data[0])
            self._model_pool.add_samples(samples)
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
        # print('[ DEBUG ]: repeat training {} times'.format(self._n_train_repeat))
        # print('[ DEBUG ]: ' + '-' * 30)
        for i in range(self._n_train_repeat):
            logs = self._do_training(
                iteration=timestep,
                batch=self._training_batch())
            log_buffer.append(logs)
        logs_buffer = {k: np.mean([item[k] for item in log_buffer]) for k in logs}


        self._num_train_steps += self._n_train_repeat
        self._train_steps_this_epoch += self._n_train_repeat
        return logs_buffer

    # HERE is the most important to revise for RNN
    def _training_batch(self, batch_size=None):
        batch_size = batch_size or self.sampler._batch_size
        env_batch_size = int(batch_size*self._real_ratio)
        model_batch_size = batch_size - env_batch_size
        # TODO: how to set teriminal state.
        # TODO: how to set model pool.

        ## can sample from the env pool even if env_batch_size == 0

        # TODO (luofm): sample trajectories (k-branch mode)
        env_batch = self._env_pool.random_batch(env_batch_size)

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
        # print('[ DEBUG ]: ' + '-' * 30)
        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._update_target()
        logs = {k: np.mean(res[1][k]) for k in res[1]}
        # for k, v in logs.items():
        #     print("[ DEBUG ] k: {}, v: {}".format(k, v))
        #     self._writer.add_scalar(k, v, iteration)
        return logs

    def _update_target(self):
        self._session.run(self.target_update)

    def _get_feed_dict(self, iteration, batch):
        """Construct TensorFlow feed_dict from sample batch."""
        state_dim = len(batch['observations'].shape)
        resize = lambda x: x[None] if state_dim == 2 else x
        # print('[ DEBUG ]: len shape: ', np.shape(np.sum(batch['valid'], axis=1).squeeze()), np.sum(batch['valid'], axis=1).squeeze())
        # TODO: need to set the
        feed_dict = {
            self._observations_ph: resize(batch['observations']),
            self._actions_ph: resize(batch['actions']),
            self._next_observations_ph: resize(batch['next_observations']),
            self._rewards_ph: resize(batch['rewards']),
            self._terminals_ph: resize(batch['terminals']),
            self._valid_ph: resize(batch['valid']),
            self.seq_len: np.sum(batch['valid'], axis=1).squeeze(),
            self._prev_state_p_ph: self.make_init_hidden(batch['observations'].shape[0])[0],
            self._prev_state_v_ph: self.make_init_hidden(batch['observations'].shape[0])[0],
            self._last_actions_ph: resize(batch['last_actions'])
        }
        # for k, v in feed_dict.items():
        #     print("{}: {}".format(k, v.shape))

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

        (Q_value1, Q_value2, Q_losses, alpha, global_step) = self._session.run(
            (self.Q1,
             self.Q2,
             self.Q_loss,
             self._alpha,
             self.global_step),
            feed_dict)
        Q_values = np.concatenate((Q_value1, Q_value2), axis=0)
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
