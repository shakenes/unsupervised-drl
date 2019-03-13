from __future__ import division
import argparse

from PIL import Image
import numpy as np
import gym

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.layers import Dense, Flatten, Permute, Input, Conv2D, Reshape
from keras.models import Model
from keras.optimizers import SGD, RMSprop, Adam
import keras.backend as K
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras import regularizers

from rl.agents.dqn import DQNAgent, DQNAgentAE, DQNAgentSFA
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from rl.powerSFA import *

from datetime import datetime
import os
import vizdoomgym

INPUT_SHAPE = (128, 128)
WINDOW_LENGTH = 1


class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize((INPUT_SHAPE[1], INPUT_SHAPE[0])).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        # return np.clip(reward, -1., 1.)
        return reward


parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='VizdoomHealthGathering-v0')
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--filters', choices=['scratch', 'gabor', 'pretrained', 'convAE', 'SFA', 'combination'], default='scratch')
parser.add_argument('--steps', type=int, default=2500000)
parser.add_argument('--freeze', type=bool, default=False)
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
nb_actions = env.action_space.n

# directory of pretrained model
save_dir = os.path.join(os.getcwd(), 'saved_models')
pretrained_model_name = 'ILSVRC-CNN-2Conv.h5'
pretrained_model_path = os.path.join(save_dir, pretrained_model_name)

# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
# Convolutional Autoencoder
if args.filters == 'convAE':
    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
    input_tensor = Input(shape=input_shape)
    if K.image_dim_ordering() == 'tf':
        # (width, height, channels)
        input_permuted = Permute((2, 3, 1))(input_tensor)
    elif K.image_dim_ordering() == 'th':
        # (channels, width, height)
        input_permuted = K.permute_dimensions(input_tensor, (1, 2, 3))
    else:
        raise RuntimeError('Unknown image_dim_ordering.')
    t = Conv2D(32, (7, 7), strides=(4, 4), activation='relu', name="conv2D_1")(input_permuted)
    t = Conv2D(32, (5, 5), strides=(2, 2), activation='relu', name="conv2D_2")(t)
    bottleneck = Flatten()(t)

    t = Dense(1024, activation='relu')(bottleneck)
    t = Dense(128 * 128, activation='sigmoid')(t)
    decoded = Reshape((1, 128, -1), name="output_decoder")(t)

    dqn_branch = Dense(1024, activation='relu', name='logic_dqn')(bottleneck)  # logic for policy
    out = Dense(nb_actions, activation='linear', name='output_actions_dqn')(dqn_branch)  # output for dqn
    model = Model(inputs=input_tensor, outputs=out)
    AEmodel = Model(inputs=input_tensor, outputs=[out, decoded])

elif args.filters == 'scratch' or args.filters == 'pretrained':
    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
    input_tensor = Input(shape=input_shape)
    if K.image_dim_ordering() == 'tf':
        # (width, height, channels)
        input_permuted = Permute((2, 3, 1))(input_tensor)
    elif K.image_dim_ordering() == 'th':
        # (channels, width, height)
        input_permuted = K.permute_dimensions(input_tensor, (1, 2, 3))
    else:
        raise RuntimeError('Unknown image_dim_ordering.')
    t = Conv2D(32, (7, 7), strides=(4, 4), activation='relu', name="conv2D_1")(input_permuted)
    t = Conv2D(32, (5, 5), strides=(2, 2), activation='relu', name="conv2D_2")(t)
    t = Flatten()(t)
    t = Dense(1024, activation='relu')(t)
    out = Dense(nb_actions, activation='linear')(t)
    model = Model(inputs=input_tensor, outputs=out)

elif args.filters == 'SFA':
    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
    input_tensor = Input(shape=input_shape)
    if K.image_dim_ordering() == 'tf':
        # (width, height, channels)
        input_permuted = Permute((2, 3, 1))(input_tensor)
    elif K.image_dim_ordering() == 'th':
        # (channels, width, height)
        input_permuted = K.permute_dimensions(input_tensor, (1, 2, 3))
    else:
        raise RuntimeError('Unknown image_dim_ordering.')
    t = Conv2D(32, (7, 7), strides=(4, 4), activation='relu', name="conv2D_1")(input_permuted)
    t = Conv2D(32, (5, 5), strides=(2, 2), activation='relu', name="conv2D_2")(t)

    t = Flatten()(t)

    slow_features = Dense(16, activation='linear')(t)

    slow_features_whitened = PowerWhitening(output_dim=8, n_iterations=50, name='whitening')(slow_features)

    t = Dense(1024, activation='relu')(t)
    out = Dense(nb_actions, activation='linear')(t)
    model = Model(inputs=input_tensor, outputs=out)

    SFAmodel = Model(inputs=input_tensor, outputs=slow_features_whitened)
    SFAmodel.compile(loss=unordered_gsfa_loss, optimizer=Adam())
    print("SFA model:")
    print(SFAmodel.summary())

elif args.filters == 'combination':
    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
    input_tensor = Input(shape=input_shape)
    if K.image_dim_ordering() == 'tf':
        # (width, height, channels)
        input_permuted = Permute((2, 3, 1))(input_tensor)
    elif K.image_dim_ordering() == 'th':
        # (channels, width, height)
        input_permuted = K.permute_dimensions(input_tensor, (1, 2, 3))
    else:
        raise RuntimeError('Unknown image_dim_ordering.')

    # convolutional encoder:
    cae_conv_1 = Conv2D(32, (7, 7), strides=(4, 4), activation='relu', name="cae_conv_1")(input_permuted)
    cae_conv_2 = Conv2D(32, (5, 5), strides=(2, 2), activation='relu', name="cae_conv_2")(cae_conv_1)
    bottleneck = Flatten()(cae_conv_2)

    # pretrained conv layers:
    pretr_conv_1 = Conv2D(32, (7, 7), strides=(4, 4), activation='relu', name="pretr_conv_1")(input_permuted)
    pretr_conv_2 = Conv2D(32, (5, 5), strides=(2, 2), activation='relu', name="pretr_conv_2")(pretr_conv_1)
    features_pretr = Flatten()(pretr_conv_2)

    t = Dense(1024, activation='relu', name='decoder_1')(bottleneck)
    t = Dense(128 * 128, activation='sigmoid', name='decoder_2')(t)
    decoded = Reshape((1, 128, -1), name="output_decoder")(t)

    pretr_cae_combined = Concatenate(axis=-1)([bottleneck, features_pretr])

    dqn_branch = Dense(1024, activation='relu', name='logic_dqn')(pretr_cae_combined)  # logic for policy
    out = Dense(nb_actions, activation='linear', name='output_actions_dqn')(dqn_branch)  # output for dqn
    model = Model(inputs=input_tensor, outputs=out)
    AEmodel = Model(inputs=input_tensor, outputs=[out, decoded])

else:
    print("unknown filter option {}".format(args.filters))
    exit()

print("DQN model:")
print(model.summary())

if args.filters == 'combination':
    tmp_model = load_model(pretrained_model_path)
    for x, y in zip([3, 5], [2,3]):
        W = tmp_model.layers[y].get_weights()
        if model.layers[x].name[0:5] != 'pretr':
            print("Wrong layer when setting pretrained weights!")
            exit()
        model.layers[x].set_weights(W)
        print("Set weights for layer {}".format(x))
        if args.freeze:
            model.layers[x].trainable = False
            print("Freeze layer")

if args.filters == 'pretrained':
    tmp_model = load_model(pretrained_model_path)
    for x in [2, 3]:
        W = tmp_model.layers[x].get_weights()
        model.layers[x].set_weights(W)
        print("Set weights for layer {}".format(x))
        if args.freeze:
            model.layers[x].trainable = False
            print("Freeze layer")

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=WINDOW_LENGTH)
processor = AtariProcessor()

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.2, value_test=.05,
                              nb_steps=125000)

# The trade-off between exploration and exploitation is difficult and an on-going research topic.
# If you want, you can experiment with the parameters or use a different policy. Another popular one
# is Boltzmann-style exploration:
# policy = BoltzmannQPolicy(tau=1.)
# Feel free to give it a try!

if args.filters == 'convAE' or args.filters == 'combination':
    dqn = DQNAgentAE(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                     processor=processor, nb_steps_warmup=50000, gamma=.999, target_model_update=10000,
                     train_interval=4, delta_clip=1., batch_size=32, test_policy=policy)
    dqn.compile(optimizer=[SGD(), Adam()], metrics=['mae'], aemodel=AEmodel)

elif args.filters == 'SFA':
    dqn = DQNAgentSFA(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                      processor=processor, nb_steps_warmup=50000, gamma=.999, target_model_update=10000,
                      train_interval=4, delta_clip=1., batch_size=32, test_policy=policy, SFAmodel=SFAmodel)
    dqn.compile(optimizer=SGD(), metrics=['mae'])
else:
    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                   processor=processor, nb_steps_warmup=50000, gamma=.999, target_model_update=10000,
                   train_interval=4, delta_clip=1., batch_size=32, test_policy=policy)
    dqn.compile(optimizer=SGD(), metrics=['mae'])

if args.mode == 'train':
    date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")

    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that you can the built-in Keras callbacks!
    weights_filename = '../../../output/' + date + 'dqn_{}_weights_{}.h5f'.format(args.env_name, args.filters)
    model_filename = '../../../output/' + date + 'dqn_{}_model_{}.h5f'.format(args.env_name, args.filters)
    checkpoint_weights_filename = '../../../output/' + date + 'dqn_' + args.env_name + '_weights_' + args.filters + '_{step}.h5f'
    log_filename = '../../../output/' + date + 'dqn_{}_log_{}.json'.format(args.env_name, args.filters)
    tb_log_dir = os.path.join('../../output/tb_logs/',
                              date + "_" +
                              args.env_name + "_" +
                              args.filters)
    tb_callback = TensorBoard(log_dir=tb_log_dir, histogram_freq=0, write_grads=False)
    callbacks = [ModelIntervalCheckpoint(os.path.join(save_dir, checkpoint_weights_filename), interval=100000)]
    callbacks += [FileLogger(os.path.join(save_dir, log_filename), interval=100)]
    callbacks += [tb_callback]
    dqn.fit(env, callbacks=callbacks, nb_steps=args.steps, log_interval=10000)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(os.path.join(save_dir, weights_filename), overwrite=True)
    model.save(os.path.join(save_dir, model_filename), overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    # dqn.test(env, nb_episodes=10, visualize=True)
elif args.mode == 'test':
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True)
