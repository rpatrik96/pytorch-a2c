import torch
import torch.nn as nn
import torch.optim as optim

from model import A2CNet

class ICMAgent(nn.Module):
    def __init__(self, n_stack, num_envs, num_actions, in_size=288, feat_size=256, lr=1e-4):
        """
        Container class of an A2C and an ICM network, the baseline for experimenting with other curiosity-based
        methods.

        :param n_stack: number of frames stacked
        :param num_envs: number of parallel environments
        :param num_actions: size of the action space of the environment
        :param in_size: dimensionality of the input tensor
        :param feat_size: number of the features
        :param lr: learning rate
        """
        super().__init__()

        # constants
        self.n_stack = n_stack
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.in_size = in_size
        self.feat_size = feat_size
        self.is_cuda = torch.cuda.is_available()

        # networks
        self.a2c = A2CNet(self.n_stack, self.num_actions, self.in_size)

        if self.is_cuda:
            self.a2c.cuda()

        # init LSTM buffers with the number of the environments
        self.a2c.set_recurrent_buffers(num_envs)

        # optimizer
        self.lr = lr
        self.optimizer = optim.Adam(self.a2c.parameters(), self.lr)
