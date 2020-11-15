import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        ## return the action that maximizes the Q-value 
        # at the current observation as the output
        observation = ptu.from_numpy(observation)
        actions = torch.argmax(self.critic.q_net_target(observation),dim=1)

        return ptu.to_numpy(actions.squeeze())