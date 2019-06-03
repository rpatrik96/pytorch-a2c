from collections import deque

import numpy as np
import torch


class RolloutStorage(object):
    def __init__(self, rollout_size, num_envs, frame_shape, n_stack, feature_size=288, is_cuda=True, value_coeff=0.5,
                 entropy_coeff=0.02, writer=None):
        """

        :param rollout_size: number of steps after the policy gets updated
        :param num_envs: number of environments to train on parallel
        :param frame_shape: shape of a frame as a tuple
        :param n_stack: number of frames concatenated
        :param is_cuda: flag whether to use CUDA
        """
        super().__init__()

        self.rollout_size = rollout_size
        self.num_envs = num_envs
        self.n_stack = n_stack
        self.frame_shape = frame_shape
        self.feature_size = feature_size
        self.is_cuda = is_cuda
        self.episode_rewards = deque(maxlen=10)

        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.writer = writer

        # initialize the buffers with zeros
        self.reset_buffers()

    def _generate_buffer(self, size):
        """
        Generates a `torch.zeros` tensor with the specified size.

        :param size: size of the tensor (tuple)
        :return:  tensor filled with zeros of 'size'
                    on the device specified by self.is_cuda
        """
        if self.is_cuda:
            return torch.zeros(size).cuda()
        else:
            return torch.zeros(size)

    def reset_buffers(self):
        """
        Creates and/or resets the buffers - each of size (rollout_size, num_envs) -
        storing: - rewards
                 - states
                 - actions
                 - log probabilities
                 - values
                 - dones

         NOTE: calling this function after a `.backward()` ensures that all data
         not needed in the future (which may `requires_grad()`) gets freed, thus
         avoiding memory leak
        :return:
        """
        self.rewards = self._generate_buffer((self.rollout_size, self.num_envs))

        # here the +1 comes from the fact that we need an initial state at the beginning of each rollout
        # which is the last state of the previous rollout
        self.states = self._generate_buffer((self.rollout_size + 1, self.num_envs, self.n_stack, *self.frame_shape))

        self.actions = self._generate_buffer((self.rollout_size, self.num_envs))
        self.log_probs = self._generate_buffer((self.rollout_size, self.num_envs))
        self.values = self._generate_buffer((self.rollout_size, self.num_envs))
        self.dones = self._generate_buffer((self.rollout_size, self.num_envs))

    def after_update(self):
        """
        Cleaning up buffers after a rollout is finished and
        copying the last state to index 0
        :return:
        """
        self.states[0].copy_(self.states[-1])
        self.actions = self._generate_buffer((self.rollout_size, self.num_envs))
        self.log_probs = self._generate_buffer((self.rollout_size, self.num_envs))
        self.values = self._generate_buffer((self.rollout_size, self.num_envs))

    def get_state(self, step):
        """
        Returns the observation of index step as a cloned object,
        otherwise torch.nn.autograd cannot calculate the gradients
        (indexing is the culprit)
        :param step: index of the state
        :return:
        """
        return self.states[step].clone()

    def obs2tensor(self, obs):
        # 1. reorder dimensions for nn.Conv2d (batch, ch_in, width, height)
        # 2. convert numpy array to _normalized_ FloatTensor
        tensor = torch.from_numpy(obs.astype(np.float32).transpose((0, 3, 1, 2))) / 255.
        return tensor.cuda() if self.is_cuda else tensor

    def insert(self, step, reward, obs, action, log_prob, value, dones):
        """
        Inserts new data into the log for each environment at index step

        :param step: index of the step
        :param reward: numpy array of the rewards
        :param obs: observation as a numpy array
        :param action: tensor of the actions
        :param log_prob: tensor of the log probabilities
        :param value: tensor of the values
        :param dones: numpy array of the dones (boolean)
        :return:
        """
        self.rewards[step].copy_(torch.from_numpy(reward))
        self.states[step + 1].copy_(self.obs2tensor(obs))
        self.actions[step].copy_(action)
        self.log_probs[step].copy_(log_prob)
        self.values[step].copy_(value)
        self.dones[step].copy_(torch.ByteTensor(dones.data))

    def _discount_rewards(self, final_value, discount=0.99):
        """
        Computes the discounted reward while respecting - if the episode
        is not done - the estimate of the final reward from that state (i.e.
        the value function passed as the argument `final_value`)


        :param final_value: estimate of the final reward by the critic
        :param discount: discount factor
        :return:
        """

        """Setup"""
        # placeholder tensor to avoid dynamic allocation with insert
        r_discounted = self._generate_buffer((self.rollout_size, self.num_envs))

        """Calculate discounted rewards"""
        # setup the reward chain
        # if the rollout has brought the env to finish
        # then we proceed with 0 as final reward (there is nothing to gain in that episode)
        # but if we did not finish, then we use our estimate

        # masked_scatter_ copies from #1 where #0 is 1 -> but we need scattering, where
        # the episode is not finished, thus the (1-x)
        R = self._generate_buffer(self.num_envs).masked_scatter((1 - self.dones[-1]).byte(), final_value)

        for i in reversed(range(self.rollout_size)):
            # the reward can only change if we are within the episode
            # i.e. while done==True, we use 0
            # NOTE: this update rule also can handle, if a new episode has started during the rollout
            # in that case an intermediate value will be 0
            # todo: add GAE
            R = self._generate_buffer(self.num_envs).masked_scatter((1 - self.dones[-1]).byte(),
                                                                    self.rewards[i] + discount * R)

            r_discounted[i] = R

        return r_discounted

    def a2c_loss(self, final_value, entropy):
        # calculate advantage
        # i.e. how good was the estimate of the value of the current state
        rewards = self._discount_rewards(final_value)
        advantage = rewards - self.values

        # weight the deviation of the predicted value (of the state) from the
        # actual reward (=advantage) with the negative log probability of the action taken
        policy_loss = (-self.log_probs * advantage.detach()).mean()

        # the value loss weights the squared difference between the actual
        # and predicted rewards
        value_loss = advantage.pow(2).mean()

        # return the a2c loss
        # which is the sum of the actor (policy) and critic (advantage) losses
        # due to the fact that batches can be shorter (e.g. if an env is finished already)
        # MEAN is used instead of SUM
        loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy

        if self.writer is not None:
            self.writer.add_scalar("a2c_loss", loss.item())
            self.writer.add_scalar("policy_loss", policy_loss.item())
            self.writer.add_scalar("value_loss", value_loss.item())
            self.writer.add_histogram("advantage", advantage.detach())
            self.writer.add_histogram("rewards", rewards.detach())
            self.writer.add_histogram("action_prob", self.log_probs.detach())

        return loss

    def log_episode_rewards(self, infos):
        """
        Logs the episode rewards

        :param infos: infos output of env.step()
        :return:
        """

        for info in infos:
            if 'episode' in info.keys():
                self.episode_rewards.append(info['episode']['r'])

    def print_reward_stats(self):
        if len(self.episode_rewards) > 1:
            print(
                "Mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".format(
                    np.mean(self.episode_rewards),
                    np.median(
                        self.episode_rewards),
                    np.min(self.episode_rewards),
                    np.max(self.episode_rewards)))
