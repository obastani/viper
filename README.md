VIPER
=====

VIPER is a tool extracting decision tree policies from an oracle (e.g., a deep Q network) using imitation learning.

Table of Contents
=====
0. Prerequisites
1. Running VIPER
2. Contact

Prerequisites
=====

VIPER requires Python 3, and the Python packages `numpy`, `tensorflow`, `scikit-learn`, `gym`, and `opencv-python` (all of which can be installed using `pip`).

Running VIPER
=====

We have included an example of how to run VIPER using the DQN oracle for the Atari Pong environment obtained from OpenAI baselines (available at `https://github.com/openai/baselines/`). To try this example, run

    $ cd python
    $ python -m viper.pong.main

Contact
=====

For questions, feel free to contact `obastani@csail.mit.edu`.
