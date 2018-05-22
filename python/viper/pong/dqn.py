# Copyright 2017-2018 MIT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import tensorflow.contrib.layers as layers

# As described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
class DQNPolicy:
    def __init__(self, env, model_path):
        # Setup
        self.env = env
        self.model_path = model_path
        self.num_actions = env.action_space.n
        self.input_shape = env.observation_space.shape
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            with tf.variable_scope('deepq'):
                # Observations
                self.imgs = tf.placeholder(tf.uint8, [None] + list(self.input_shape), name='observation')

                # Randomness
                self.stochastic_ph = tf.placeholder(tf.bool, (), name='stochastic')
                self.update_eps_ph = tf.placeholder(tf.float32, (), name='update_eps')
                eps = tf.get_variable('eps', (), initializer=tf.constant_initializer(0))

                # Q-function
                with tf.variable_scope('q_func'):
                    # Normalization
                    out = tf.cast(self.imgs, tf.float32) / 255.0

                    # Convolutions
                    with tf.variable_scope('convnet'):
                        out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
                        out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
                        out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)

                    # Flatten
                    conv_out = layers.flatten(out)

                    # Fully connected
                    with tf.variable_scope('action_value'):
                        value_out = layers.fully_connected(conv_out, num_outputs=512, activation_fn=None)
                        value_out = tf.nn.relu(value_out)
                        value_out = layers.fully_connected(value_out, num_outputs=self.num_actions, activation_fn=None)

                # Q values
                self.qs = value_out

                # Deterministic actions
                deterministic_actions = tf.argmax(self.qs, axis=1)

                # Stochastic actions
                batch_size = tf.shape(self.imgs)[0]
                random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=self.num_actions, dtype=tf.int64)
                chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
                stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

                # Output actions
                self.output_actions = tf.cond(self.stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
                self.update_eps_expr = eps.assign(tf.cond(self.update_eps_ph >= 0, lambda: self.update_eps_ph, lambda: eps))

            # Load
            tf.train.Saver().restore(self.sess, self.model_path)

    def predict_q(self, imgs):
        with self.graph.as_default():
            # Inputs
            feed_dict = {}
            feed_dict[self.imgs] = imgs
            feed_dict[self.update_eps_ph] = -1.0
            feed_dict[self.stochastic_ph] = False

            # Q values
            qs = self.sess.run(self.qs, feed_dict=feed_dict)

            return qs
    
    def predict(self, imgs):
        with self.graph.as_default():
            # Inputs
            feed_dict = {}
            feed_dict[self.imgs] = imgs
            feed_dict[self.update_eps_ph] = -1.0
            feed_dict[self.stochastic_ph] = False

            # Action
            acts = self.sess.run(self.output_actions, feed_dict=feed_dict)

            # Updates
            self.sess.run(self.update_eps_expr, feed_dict=feed_dict)

            return acts
