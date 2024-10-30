from gym.vector.utils import spaces
import numpy as np

from computations import body_ang_vel_pqr, DEG2RAD, RAD2DEG
from fly_env import FlyEnv
import utils


class DiscreteFlyEnv(FlyEnv):
    def __init__(self, config_path, state_space, action_space):
        super(DiscreteFlyEnv, self).__init__(config_path)

        self.observation_space = spaces.MultiDiscrete(state_space)
        self.action_space = spaces.Discrete(action_space)

        # Set ranges for the various parameters of the state and actions
        self.pitch_range = (-180 * DEG2RAD, 180 * DEG2RAD)
        self.delta_pitch_range = (-15, 15)
        self.delta_phi_range = (-0.1, 0.1)

    def step(self, action):
        # Translate from discrete action space to continuous action space
        delta_phi = utils.map_range(action, 0, self.action_space.n-1, self.delta_phi_range[0], self.delta_phi_range[1])
        continuous_action = np.array([0, 0, delta_phi, 0, 0, delta_phi, 0])

        # Discretize observation
        obs, reward, is_done, info = super(DiscreteFlyEnv, self).step(action=continuous_action)
        
        pitch = obs.euler_angles[1]
        discrete_pitch = int(round(utils.map_range(pitch, self.pitch_range[0], self.pitch_range[1], 0, self.observation_space.nvec[0] - 1)))
        delta_pitch = body_ang_vel_pqr(obs.euler_angles, obs.body_angular_vel, False)[1]
        delta_pitch = np.clip([delta_pitch], a_min=self.delta_pitch_range[0], a_max=self.delta_pitch_range[1])[0]
        discrete_delta_pitch = int(round(utils.map_range(delta_pitch, self.delta_pitch_range[0], self.delta_pitch_range[1], 0, self.observation_space.nvec[1] - 1)))

        # Compute reward
        target_angle = -45 * DEG2RAD
        dist_sq_on_circle = (np.cos(pitch) - np.cos(target_angle))**2 + (np.sin(pitch) - np.sin(target_angle))**2
        reward = 20 * np.exp(-5 * dist_sq_on_circle) - 0.01 * (delta_pitch ** 2)

        return [discrete_pitch, discrete_delta_pitch], reward, is_done, info


    def reset(self):
        state = super(DiscreteFlyEnv, self).reset()
        pitch = utils.standardize_angle(state.euler_angles[1])
        delta_pitch = body_ang_vel_pqr(state.euler_angles, state.body_angular_vel, False)[1]
        delta_pitch = np.clip([delta_pitch], a_min=self.delta_pitch_range[0], a_max=self.delta_pitch_range[1])[0]
        discrete_pitch = int(round(utils.map_range(pitch, self.pitch_range[0], self.pitch_range[1], 0, self.observation_space.nvec[0] - 1)))
        discrete_delta_pitch = int(round(utils.map_range(delta_pitch, self.delta_pitch_range[0], self.delta_pitch_range[1], 0, self.observation_space.nvec[1] - 1)))
        return [discrete_pitch, discrete_delta_pitch]