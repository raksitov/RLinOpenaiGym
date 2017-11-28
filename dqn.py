import gym
import random
import itertools
import shutil
import os

import logging
import numpy as np
import tensorflow as tf

from tensorflow.contrib import layers
from math import log10
from time import time
from gym import wrappers
from collections import namedtuple
from pprint import pprint
from enum import Enum

# Environment.
ENVIRONMENT = 'CartPole-v0'
STEPS = 200
TARGET_REWARD = 195
MAX_DURATION = 200

# Auxiliary.
AVERAGING_INTERVAL = 100
EPISODES = 100000
SUMMARY_DIR = ENVIRONMENT
TRIALS = 1
VERBOSE = False

# Hparams.
EPSILON_START = 0.5
EPSILON_END = 0.01
BATCH_SIZE = 32
UPDATE_FREQUENCY = 1
LEARNING_RATE_START = 0.001
LEARNING_RATE_END = 0.000001
GAMMA = 0.9
REPLAY_MEMORY_SIZE = 10000
DEEP_NETWORK = [64]
RANDOM_PLAY = 100
CLIP_BY_NORM = 0

# Target network.
UPDATE_TARGET_NETWORK_COUNT = 100
UPDATE_TARGET_NETWORK_RATE = 0.01

Properties = namedtuple(
    'Properties',
    ['replay', 'target_network', 'continuous_update', 'double_network'])

State = namedtuple(
    'State', ['observation', 'action', 'new_observation', 'reward', 'done'])


def mprint(string):
  if VERBOSE:
    print string


def linear_decay(start, end, average_reward):
  return end + (start - end) * (1 - average_reward)


class Model:

  def __init__(self,
               session,
               observation_size,
               num_actions,
               scope,
               summaries_dir=None):
    self.scope = scope
    with tf.variable_scope(scope):
      self.counter = 0
      self.session = session
      self.num_actions = num_actions
      self.network_structure = DEEP_NETWORK
      self.states = tf.placeholder(tf.float32, shape=(None, observation_size))
      self.labels = tf.placeholder(tf.float32, shape=(None,))
      self.actions = tf.placeholder(tf.int32, shape=(None,))
      self.average_reward = tf.placeholder(tf.float32, shape=())
      self.learning_rate = tf.placeholder(tf.float32, shape=())

      self._build_network()
      if summaries_dir is not None:
        tf.summary.scalar('Average Reward', self.average_reward * TARGET_REWARD)
        self.merged_summary = tf.summary.merge_all()
        if os.path.exists(summaries_dir):
          shutil.rmtree(summaries_dir)
        self.summary_writer = tf.summary.FileWriter(summaries_dir,
                                                    self.session.graph)

  def _minimize(self, optimizer):
    if CLIP_BY_NORM:
      gradients = optimizer.compute_gradients(self.loss)
      for i, (gradient, variable) in enumerate(gradients):
        if gradient is not None:
          gradients[i] = (tf.clip_by_norm(gradient, CLIP_BY_NORM), variable)
      return optimizer.apply_gradients(gradients)
    else:
      return optimizer.minimize(self.loss)

  def _build_network(self):
    last_layer = layers.fully_connected(self.states, self.network_structure[0])
    for i in xrange(1, len(self.network_structure)):
      last_layer = layers.fully_connected(last_layer, self.network_structure[i])
    self.predictions = layers.fully_connected(last_layer, self.num_actions)

    self.q_values = tf.reduce_sum(
        tf.multiply(self.predictions, tf.one_hot(self.actions,
                                                 self.num_actions)),
        reduction_indices=1)
    self.loss = tf.reduce_sum(tf.square(self.labels - self.q_values))
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=self.learning_rate)
    self.train_op = self._minimize(optimizer)

  def train(self, observations, rewards, actions, average_reward):
    self.counter += 1
    summary, loss, _ = self.session.run(
        [self.merged_summary, self.loss, self.train_op],
        feed_dict={
            self.states:
                observations,
            self.labels:
                rewards,
            self.actions:
                actions,
            self.average_reward:
                average_reward,
            self.learning_rate:
                linear_decay(LEARNING_RATE_START, LEARNING_RATE_END,
                             average_reward),
        })
    if self.summary_writer is not None:
      if self.counter % 10 == 0:
        self.summary_writer.add_summary(summary, self.counter)
      if self.counter % 100 == 0:
        mprint('Batch: {}, loss: {}, average reward: {}'.format(
            self.counter, loss, average_reward * TARGET_REWARD))

  def predict(self, observation):
    predictions = self.session.run(
        [self.predictions], feed_dict={
            self.states: [observation],
        })

    return predictions

  def cleanup(self):
    self.summary_writer.close()


class TargetUpdater():

  def __init__(self, session, main_model, target_model, update_rate):
    self.session = session
    params1 = [
        t for t in tf.trainable_variables()
        if t.name.startswith(main_model.scope)
    ]
    params1.sort(key=lambda v: v.name)
    params2 = [
        t for t in tf.trainable_variables()
        if t.name.startswith(target_model.scope)
    ]
    params2.sort(key=lambda v: v.name)
    self.copy_ops = []
    for v1, v2 in zip(params1, params2):
      op = v2.assign(v1.value() * update_rate + v2.value() * (1 - update_rate))
      self.copy_ops.append(op)

  def update_target(self):
    self.session.run(self.copy_ops)


class Agent:

  def __init__(self, observation_size, num_actions, properties):
    self.properties = properties
    self._ensure_correctness()
    self.num_actions = num_actions
    self.states = []
    tf.reset_default_graph()
    session = tf.Session()
    self.main_network = Model(session, observation_size, num_actions,
                              'Main_network', self._get_summary_dir())
    self.target_network = Model(session, observation_size, num_actions,
                                'Target_network')
    self.target_updater = TargetUpdater(
        session,
        self.main_network,
        self.target_network,
        update_rate=UPDATE_TARGET_NETWORK_RATE
        if properties.continuous_update else 1)
    session.run(tf.global_variables_initializer())
    # Update target to the same initialization that main model got.
    self.target_updater.update_target()
    self.counter = 0

  def _get_summary_dir(self):
    encoded = ''.join([str(int(prop)) for prop in self.properties])
    return '{}-{}'.format(SUMMARY_DIR, encoded)

  def _ensure_correctness(self):
    if self.properties.double_network or self.properties.continuous_update:
      assert self.properties.target_network

  def add_state(self, state):
    self.states.append(state)
    pass

  def _recent_memories(self):
    start = 0
    if len(self.states) >= REPLAY_MEMORY_SIZE:
      start = -REPLAY_MEMORY_SIZE
    return self.states[start:]

  def _get_batch(self):
    if (len(self.states) < BATCH_SIZE or
        len(self.states) % UPDATE_FREQUENCY != 0):
      return None
    if self.properties.replay:
      return random.sample(self._recent_memories(), BATCH_SIZE)
    return self.states[-BATCH_SIZE:]

  def _target_network_need_update(self):
    if not self.properties.target_network:
      return False
    if not self.properties.continuous_update:
      return self.counter % UPDATE_TARGET_NETWORK_COUNT == 0
    return True

  def train(self, average_reward):
    batch = self._get_batch()
    if not batch:
      return
    observations = []
    rewards = []
    actions = []
    for sample in batch:
      observations.append(sample.observation)
      reward = sample.reward
      if not sample.done:
        reward += GAMMA * self._Q(sample)
      rewards.append(reward)
      actions.append(sample.action)
    self.main_network.train(observations, rewards, actions, average_reward)
    self.counter += 1
    if self._target_network_need_update():
      self.target_updater.update_target()

  def _Q(self, sample):
    predictions = self._get_predict_network().predict(sample.new_observation)
    if self.properties.double_network:
      action = self.choose_action(sample.new_observation)
      return predictions[0][0][action]
    return np.max(predictions)

  def _get_predict_network(self):
    if self.properties.target_network:
      return self.target_network
    return self.main_network

  def choose_action(self, observation):
    return np.argmax(self.main_network.predict(observation))

  def cleanup(self):
    self.main_network.cleanup()
    self.target_network.cleanup()


def choose_action(observation, agent, num_actions, average_reward, episode):
  action = None
  if episode < RANDOM_PLAY or random.random() < linear_decay(
      EPSILON_START, EPSILON_END, average_reward):
    action = np.random.randint(0, num_actions)
  else:
    action = agent.choose_action(observation)
  return action


def run_experiment(properties, retries=0):
  env = gym.make(ENVIRONMENT)
  num_actions = env.action_space.n
  agent = Agent(env.observation_space.shape[0], num_actions, properties)

  average_reward = 0
  results = []
  start_time = time()
  for episode in xrange(EPISODES):
    observation = env.reset()
    total_reward = 0

    if len(results) >= AVERAGING_INTERVAL:
      average_reward = np.mean(
          results[-AVERAGING_INTERVAL:]) / float(TARGET_REWARD)
      if episode % 100 == 0:
        mprint(
            'Average reward for last 100 episodes: {}'.format(average_reward))
      duration = time() - start_time
      if average_reward >= 1.:
        print('Done after {} episodes in {} seconds'.format(episode, duration))
        return average_reward, episode, duration
      elif duration > MAX_DURATION:
        print('Timeout after {} episodes'.format(episode))
        if retries:
          print('Re-running')
          return run_experiment(properties, retries - 1)
        return average_reward, episode, duration

    for step in xrange(STEPS):
      action = choose_action(observation, agent, num_actions, average_reward,
                             episode)
      new_observation, reward, done, _ = env.step(action)
      agent.add_state(State(observation, action, new_observation, reward, done))
      observation = new_observation

      agent.train(average_reward)

      total_reward += reward
      if done:
        if episode % 100 == 0:
          mprint('Episode - {}'.format(episode))
        results.append(total_reward)
        break
  agent.cleanup()
  return average_reward, EPISODES, time() - start_time


def all_modes():
  for replay in [False, True]:
    for target_network in [False, True]:
      for continuous_update in [False, True]:
        for double_network in [False, True]:
          if (continuous_update or double_network) and not target_network:
            continue
          yield Properties(
              replay=replay,
              target_network=target_network,
              continuous_update=continuous_update,
              double_network=double_network)


def run_comparison():
  for properties in all_modes():
    print properties
    pprint(run_experiment(properties))


def main():
  run_comparison()


if __name__ == '__main__':
  main()
