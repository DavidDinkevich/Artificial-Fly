
import numpy as np

from fly_env import FlyEnvState
from computations import body_ang_vel_pqr, DEG2RAD


class FlyEnvController:
    '''
    A Controller class for a FlyEnv simulation.
    The job of a Controller is to output an action given a state of the simulation.
    '''
    def respond(self, state: FlyEnvState) -> np.ndarray:
        pass


class DiscreteFlyEnvController:
    '''
    A Controller class for a DiscreteFlyEnv simulation.
    The job of a Controller is to output an action given a state of the simulation.
    A state is an ndarray of the following structure: [delta_phi_left, delta_phi_right]
    '''
    def respond(self, state: np.ndarray) -> np.ndarray:
        pass


class PitchStabilizerController(FlyEnvController):
    '''
    This controller is taken from the matlab version of the simulation.
    Stabilizes pitch to -45 degrees.
    '''
    
    def __init__(self):
        self.Kp, self.Ki = 8 / 1000, 0.5 # Constants taken from Whitehead paper (see paper or matlab version)

    def respond(self, state: FlyEnvState) -> np.ndarray:
        euler_dot_for_controller = body_ang_vel_pqr(state.euler_angles, state.body_angular_vel, False)
        theta = state.euler_angles[1]  # Theta (pitch)
        delta_phi = euler_dot_for_controller[1] * self.Kp + (theta - (-45 * DEG2RAD)) * self.Ki
        return np.array([0, 0, delta_phi, 0, 0, delta_phi, 0])


class HeightStabilizerController(FlyEnvController):
    '''
    This controller both stabilizes pitch, in addition to trying to keep the fly at a desired height.
    '''

    def __init__(self, desired_height=0, max_dev=10):
        self.pitch_controller = PitchStabilizerController()
        self.max_error = max_dev
        self.desired_height = desired_height + self.max_error
        self.baseline_bps = 235

    def respond(self, state: FlyEnvState):
        action = self.pitch_controller.respond(state)

        curr_height = state.loc[2] * 1000
        height_error = self.desired_height - curr_height
        min_bps, max_bps = 200, 250

        alpha = min(self.max_error, abs(height_error)) / self.max_error
        bps = min_bps + alpha * (max_bps - min_bps)
        action[6] = bps - state.bps

        return action


class QController(DiscreteFlyEnvController):
    '''
    This controller responds according to a given Q-Table.
    '''
    
    def __init__(self, q_table):
        self.action_table = np.argmax(q_table, axis=-1)

    def respond(self, state: np.ndarray) -> np.ndarray:
        return np.array([self.action_table[*state]])
