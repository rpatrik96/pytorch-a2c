from pdb import set_trace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def init(module, weight_init, bias_init, gain=1):
    """

    :param module: module to initialize
    :param weight_init: initialization scheme
    :param bias_init: bias initialization scheme
    :param gain: gain for weight initialization
    :return: initialized module
    """
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class ConvBlock(nn.Module):

    def __init__(self, ch_in=4):
        """
        A basic block of convolutional layers,
        consisting: - 4 Conv2d
                    - LeakyReLU (after each Conv2d)
                    - currently also an AvgPool2d (I know, a place for me is reserved in hell for that)

        :param ch_in: number of input channels, which is equivalent to the number
                      of frames stacked together
        """
        super().__init__()

        # constants
        self.num_filter = 32
        self.size = 3
        self.stride = 2
        self.pad = self.size // 2

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('leaky_relu'))
        # layers
        self.conv1 = init_(nn.Conv2d(ch_in, self.num_filter, self.size, self.stride, self.pad))
        self.conv2 = init_(nn.Conv2d(self.num_filter, self.num_filter, self.size, self.stride, self.pad))
        self.conv3 = init_(nn.Conv2d(self.num_filter, self.num_filter, self.size, self.stride, self.pad))
        self.conv4 = init_(nn.Conv2d(self.num_filter, self.num_filter, self.size, self.stride, self.pad))

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))

        x = nn.AvgPool2d(2)(x)  # needed as the input image is 84x84, not 42x42
        # return torch.flatten(x)
        # set_trace()
        return x.view(x.shape[0], -1)  # retain batch size


class FeatureEncoderNet(nn.Module):
    def __init__(self, n_stack, in_size, is_lstm=True):
        """
        Network for feature encoding

        :param n_stack: number of frames stacked beside each other (passed to the CNN)
        :param in_size: input size of the LSTMCell if is_lstm==True else it's the output size
        :param is_lstm: flag to indicate wheter an LSTMCell is included after the CNN
        """
        super().__init__()
        # constants
        self.in_size = in_size
        self.h1 = 288
        self.is_lstm = is_lstm  # indicates whether the LSTM is needed

        # layers
        self.conv = ConvBlock(ch_in=n_stack)
        if self.is_lstm:
            self.lstm = nn.LSTMCell(input_size=self.in_size, hidden_size=self.h1)

    def reset_lstm(self, buf_size=None, reset_indices=None):
        """
        Resets the inner state of the LSTMCell

        :param reset_indices: boolean list of the indices to reset (if True then that column will be zeroed)
        :param buf_size: buffer size (needed to generate the correct hidden state size)
        :return:
        """
        if self.is_lstm:
            with torch.no_grad():
                if reset_indices is None:
                    # set device to that of the underlying network
                    # (it does not matter, the device of which layer is queried)
                    self.h_t1 = self.c_t1 = torch.zeros(buf_size, self.h1, device=self.lstm.weight_ih.device)
                else:
                    # set device to that of the underlying network
                    # (it does not matter, the device of which layer is queried)
                    resetTensor = torch.as_tensor(reset_indices.astype(np.uint8), device=self.lstm.weight_ih.device)

                    if resetTensor.sum():
                        self.h_t1 = (1 - resetTensor.view(-1, 1)).float() * self.h_t1
                        self.c_t1 = (1 - resetTensor.view(-1, 1)).float() * self.c_t1

    def forward(self, x):
        """
        In: [s_t]
            Current state (i.e. pixels) -> 1 channel image is needed

        Out: phi(s_t)
            Current state transformed into feature space

        :param x: input data representing the current state
        :return:
        """
        x = self.conv(x)

        # return self.lin(x)

        if self.is_lstm:
            x = x.view(-1, self.in_size)
            # set_trace()
            self.h_t1, self.c_t1 = self.lstm(x, (self.h_t1, self.c_t1))  # h_t1 is the output
            return self.h_t1  # [:, -1, :]#.reshape(-1)

        else:
            return x.view(-1, self.in_size)




class A2CNet(nn.Module):
    def __init__(self, n_stack, num_actions, in_size=288, writer=None):
        """
        Implementation of the Advantage Actor-Critic (A2C) network

        :param n_stack: number of frames stacked
        :param num_actions: size of the action space, pass env.action_space.n
        :param in_size: input size of the LSTMCell of the FeatureEncoderNet
        """
        super().__init__()

        self.writer = writer

        # constants
        self.in_size = in_size  # in_size
        self.num_actions = num_actions

        # networks
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.feat_enc_net = FeatureEncoderNet(n_stack, self.in_size)
        self.actor = init_(nn.Linear(self.feat_enc_net.h1, self.num_actions))  # estimates what to do
        self.critic = init_(nn.Linear(self.feat_enc_net.h1,
                                      1))  # estimates how good the value function (how good the current state is)

    def set_recurrent_buffers(self, buf_size):
        """
        Initializes LSTM buffers with the proper size,
        should be called after instatiation of the network.

        :param buf_size: size of the recurrent buffer
        :return:
        """
        self.feat_enc_net.reset_lstm(buf_size=buf_size)

    def reset_recurrent_buffers(self, reset_indices):
        """

        :param reset_indices: boolean numpy array containing True at the indices which
                              should be reset
        :return:
        """
        self.feat_enc_net.reset_lstm(reset_indices=reset_indices)

    def forward(self, state):
        """

        feature: current encoded state

        :param state: current state
        :return:
        """

        # encode the state
        feature = self.feat_enc_net(state)

        # calculate policy and value function
        policy = self.actor(feature)
        value = self.critic(feature)

        if self.writer is not None:
            self.writer.add_histogram("feature", feature.detach())
            self.writer.add_histogram("policy", policy.detach())
            self.writer.add_histogram("value", value.detach())

        return policy, torch.squeeze(value), feature

    def get_action(self, state):
        """
        Method for selecting the next action

        :param state: current state
        :return: tuple of (action, log_prob_a_t, value)
        """

        """Evaluate the A2C"""
        policy, value, feature = self(state)  # use A3C to get policy and value

        """Calculate action"""
        # 1. convert policy outputs into probabilities
        # 2. sample the categorical  distribution represented by these probabilities
        action_prob = F.softmax(policy, dim=-1)
        cat = Categorical(action_prob)
        action = cat.sample()

        return (action, cat.log_prob(action), cat.entropy().mean(), value,
                feature)
