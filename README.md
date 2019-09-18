# Boosting Reinforcement Learning with Unsupervised Feature Extraction

This repository contains the code to reproduce the results presented in the paper [Boosting Reinforcement Learning with Unsupervised Feature Extraction](https://doi.org/10.1007/978-3-030-30487-4_43) presented at ICANN 2019.

The project is based on [keras-rl](https://github.com/keras-rl/keras-rl) by Matthias Plappert. We kindly thank him and all contributors for their great work.

## Installation
The code can be installed by cloning this repository and executing `pip install -e .` in the directory containing the `setup.py`.

It is also necessary to install [vizdoomgym](https://github.com/shakenes/vizdoomgym). Make sure to check out the `healthgathering` branch to use our modified reward function for Health Gathering and Health Gathering Supreme.

## Usage
Start training a DQN agent in the Basic scenario with randomly initilized filters for 1.5M time steps by executing 
`python dqn.py --env-name=VizdoomBasic-v0 --filters=scratch --steps=1500000` in the examples directory.

The environments from the paper are `VizdoomBasic-v0`, `VizdoomDefendCenter-v0`, `VizdoomHealthGathering-v0`, `VizdoomHealthGatheringSupreme-v0` and `VizdoomMyWayHome-v0`. Please refer to the documentation of [vizdoomgym](https://github.com/shakenes/vizdoomgym) for more detailed information.

The filter option determines the feature extraction method that is used. Possible options are `scratch`, `pretrained`, `convAE`, `SFA` and `combination`.
