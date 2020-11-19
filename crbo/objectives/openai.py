#  Confidence Region Bayesian Optimization -- Reference Implementation
#  Copyright (c) 2020 Robert Bosch GmbH
# 
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
# 
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
# 
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.


import gym
import numpy as np
import os

from gym.envs.classic_control.pendulum import angle_normalize
from stable_baselines.ppo2 import PPO2
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from crbo.objectives.base import ObjectiveFunction
from crbo import util


class CenterPositionReward(gym.RewardWrapper):
    def __init__(self, env):
        # TODO: How can I access the environment directly? env.env does not feel right...
        assert isinstance(env.env, gym.envs.classic_control.CartPoleEnv)
        super().__init__(env)

    def reward(self, reward):
        return np.clip(1 - np.abs(self.state[0] - 0.0), 0, np.inf)


class CartPoleObjective(ObjectiveFunction):
    def __init__(self, policy_domain_size, noise_var=0.0):
        dim =  130
        self.env_id = "CartPole-v1"
        self.__name__ = self.env_id
        model_path = f"models/{self.env_id}.pkl"
        load = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
        self.env = DummyVecEnv([lambda: CenterPositionReward(gym.make(self.env_id))])
        self.model = PPO2.load(load, env=self.env)
        self.env._max_episode_steps = 500

        self.theta0 = util.mlp_model_to_theta(self.model)
        lb, ub = self.theta0 - policy_domain_size, self.theta0 + policy_domain_size
        super(CartPoleObjective, self).__init__(
            dim=dim, x_opt=None, f_opt=1.0, noise_var=noise_var, lb=lb, ub=ub
        )

    def _fun(self, x):
        ret_val = np.zeros((x.shape[0], 1))
        for i, xi in enumerate(x):
            self.model = util.theta_to_mlp_model(np.atleast_2d(xi), self.model)
            ret_val[i] = np.sum(_run_episode(self.env, self.model)) / self.env._max_episode_steps
        return -1 * ret_val


class SmallAngleReward(gym.RewardWrapper):
    def __init__(self, env):
        assert isinstance(env.env, gym.envs.classic_control.PendulumEnv)
        super().__init__(env)

    def reward(self, reward):
        return np.clip(1 - np.abs(angle_normalize(self.state[0])) / np.deg2rad(2), 0, np.inf)


class PendulumObjective(ObjectiveFunction):
    def __init__(self, policy_domain_size, noise_var=0.0):
        dim = 65
        self.env_id = "Pendulum-v0"
        self.__name__ = self.env_id
        model_path = f"models/{self.env_id}.pkl"
        load = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
        self.env = DummyVecEnv([lambda: SmallAngleReward(gym.make(self.env_id))])
        self.model = PPO2.load(load, env=self.env)
        self.env._max_episode_steps = 200

        self.theta0 = util.mlp_model_to_theta(self.model)
        lb, ub = self.theta0 - policy_domain_size, self.theta0 + policy_domain_size
        super(PendulumObjective, self).__init__(
            dim=dim, x_opt=None, f_opt=1.0, noise_var=noise_var, lb=lb, ub=ub
        )

    def _fun(self, x):
        ret_val = np.zeros((x.shape[0], 1))
        for i, xi in enumerate(x):
            self.model = util.theta_to_mlp_model(np.atleast_2d(xi), self.model)
            rewards = _run_episode(self.env, self.model, render=False)
            episode_offset = 0
            episode_return = np.sum(rewards[episode_offset:]) / (
                self.env._max_episode_steps - episode_offset
            )
            ret_val[i] = episode_return
        return -1 * ret_val


def run_cartpole():
    f = CartPoleObjective(policy_domain_size=10)
    theta0 = f.theta0
    f0 = f(np.tile(theta0, [5, 1]))
    print(f"Mean: {np.mean(f0):.3f} +/- {np.std(f0):.3f} std")
    print(f0)


def run_pendulum():
    f = PendulumObjective(policy_domain_size=10)
    theta0 = f.theta0
    f0 = f(np.tile(theta0, [5, 1]))
    print(f"Mean: {np.mean(f0):.3f} +/- {np.std(f0):.3f} std")
    print(f0)


def _run_episode(env, model, render=False):
    obs = np.zeros((1,) + env.observation_space.shape)
    obs[:], done = env.reset(), False
    rewards = []

    while not done:
        actions = model.predict(obs, deterministic=True)[0]
        obs[:], reward, done, _ = env.step(actions)
        rewards.append(reward)
        env.render("human") if render else None
    return np.array(rewards)


if __name__ == "__main__":
    run_cartpole()
    run_pendulum()
