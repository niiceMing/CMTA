# This code is taken from: https://raw.githubusercontent.com/rlworkgroup/garage/af57bf9c6b10cd733cb0fa9bfe3abd0ba239fd6e/src/garage/envs/normalized_env.py
#
# """"An environment wrapper that normalizes action, observation and reward."""
# type: ignore
import gym
import gym.spaces
import gym.spaces.utils
import numpy as np
import random
import pickle
import pdb

class ResetTaskEnvWrapper(gym.Wrapper):
    """An environment wrapper for reset metaworld env task.

    This wrapper normalizes action, and optionally observation and reward.

    Args:
        env (garage.envs.GarageEnv): An environment instance.
        scale_reward (float): Scale of environment reward.
        normalize_obs (bool): If True, normalize observation.
        normalize_reward (bool): If True, normalize reward. scale_reward is
            applied after normalization.
        expected_action_scale (float): Assuming action falls in the range of
            [-expected_action_scale, expected_action_scale] when normalize it.
        flatten_obs (bool): Flatten observation if True.
        obs_alpha (float): Update rate of moving average when estimating the
            mean and variance of observations.
        reward_alpha (float): Update rate of moving average when estimating the
            mean and variance of rewards.

    """

    def __init__(
        self,
        env,
        train_task,
        mode,
    ):
        super().__init__(env)

        # self.env = env
        if mode == 'train':
            # self.train_task = train_task[0:10]
            self.train_task = train_task
        else:
            # self.train_task = train_task[0:10]
            # self.train_task = train_task[40:50]    
            self.train_task = train_task
    
    def reset(self, **kwargs):
        """Reset environment.

        Args:
            **kwargs: Additional parameters for reset.

        Returns:
            tuple:
                * observation (np.ndarray): The observation of the environment.
                * reward (float): The reward acquired at this time step.
                * done (boolean): Whether the environment was completed at this
                    time step.
                * infos (dict): Environment-dependent additional information.

        """
        task = random.choice(self.train_task)   
 
        self.env.set_task(task)     
        ret = self.env.reset(**kwargs)
        return ret
