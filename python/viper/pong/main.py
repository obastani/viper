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

from ..core.rl import *
from .pong import *
from .dqn import *
from ..core.dt import *
from ..util.log import *

def learn_dt():
    # Parameters
    log_fname = '../pong_dt.log'
    model_path = '../data/model-atari-pong-1/saved'
    max_depth = 12
    n_batch_rollouts = 10
    max_samples = 200000
    max_iters = 80
    train_frac = 0.8
    is_reweight = True
    n_test_rollouts = 50
    save_dirname = '../tmp/pong'
    save_fname = 'dt_policy.pk'
    save_viz_fname = 'dt_policy.dot'
    is_train = True
    
    # Logging
    set_file(log_fname)
    
    # Data structures
    env = get_pong_env()
    teacher = DQNPolicy(env, model_path)
    student = DTPolicy(max_depth)
    state_transformer = get_pong_symbolic

    # Train student
    if is_train:
        student = train_dagger(env, teacher, student, state_transformer, max_iters, n_batch_rollouts, max_samples, train_frac, is_reweight, n_test_rollouts)
        save_dt_policy(student, save_dirname, save_fname)
        save_dt_policy_viz(student, save_dirname, save_viz_fname)
    else:
        student = load_dt_policy(save_dirname, save_fname)

    # Test student
    rew = test_policy(env, student, state_transformer, n_test_rollouts)
    log('Final reward: {}'.format(rew), INFO)
    log('Number of nodes: {}'.format(student.tree.tree_.node_count), INFO)

def bin_acts():
    # Parameters
    seq_len = 10
    n_rollouts = 10
    log_fname = 'pong_options.log'
    model_path = 'model-atari-pong-1/saved'
    
    # Logging
    set_file(log_fname)
    
    # Data structures
    env = get_pong_env()
    teacher = DQNPolicy(env, model_path)

    # Action sequences
    seqs = get_action_sequences(env, teacher, seq_len, n_rollouts)

    for seq, count in seqs:
        log('{}: {}'.format(seq, count), INFO)

def print_size():
    # Parameters
    dirname = 'results/run9'
    fname = 'dt_policy.pk'

    # Load decision tree
    dt = load_dt_policy(dirname, fname)

    # Size
    print(dt.tree.tree_.node_count)

if __name__ == '__main__':
    learn_dt()
