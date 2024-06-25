
import numpy as np

from fly_env import FlyEnvState
from computations import body_ang_vel_pqr, DEG2RAD


class Controller:
    def respond(self, state: np.ndarray) -> np.ndarray:
        pass


class PitchStabilizerController(Controller):
    def __init__(self):
        self.Kp, self.Ki = 8 / 1000, 0.5

    def respond(self, state: FlyEnvState) -> np.ndarray:
        euler_dot_for_controller = body_ang_vel_pqr(state.euler_angles, state.body_angular_vel, False)
        theta = state.euler_angles[1]  # Theta (pitch)
        delta_phi = euler_dot_for_controller[1] * self.Kp + (theta - (-45 * DEG2RAD)) * self.Ki
        return np.array([0, 0, delta_phi, 0, 0, delta_phi, 0])


class HeightStabilizerController(Controller):
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
