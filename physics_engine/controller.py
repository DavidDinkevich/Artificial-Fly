
import numpy as np

from fly_env import FlyEnvState
from computations import body_ang_vel_pqr, DEG2RAD


class Controller:
    def respond(self, state: FlyEnvState) -> (np.ndarray, int):
        pass


class PitchStabilizerController(Controller):
    def __init__(self):
        self.Kp, self.Ki = 8 / 1000, 0.5

    def respond(self, state: FlyEnvState) -> (np.ndarray, int):
        euler_dot_for_controller = body_ang_vel_pqr(state.euler_angles, state.body_angular_vel, False)
        theta = state.euler_angles[1]  # Theta (pitch)
        delta_phi = euler_dot_for_controller[1] * self.Kp + (theta - (-45 * DEG2RAD)) * self.Ki
        d_euler_angles = np.array([0, 0, delta_phi])
        return d_euler_angles, d_euler_angles, 0


class HeightStabilizerController(Controller):
    def __init__(self, desired_height=0):
        self.pitch_controller = PitchStabilizerController()
        self.max_error = 10
        self.desired_height = desired_height + self.max_error
        self.baseline_bps = 235
        self.bps = self.baseline_bps

    def respond(self, state: FlyEnvState):
        d_euler_angles_left, d_euler_angles_right, _ = self.pitch_controller.respond(state)

        curr_height = state.loc[2] * 1000
        # curr_height = state.loc[2] * 1000
        # print(f'curr_height = {curr_height}')
        height_error = self.desired_height - curr_height
        min_bps, max_bps = 200, 250

        alpha = min(self.max_error, abs(height_error)) / self.max_error
        self.bps = min_bps + alpha * (max_bps - min_bps)

        # alpha = 0.1
        #
        # error = max(0, np.sign(height_error))
        # self.d_bps = self.d_bps * alpha + (1-alpha) * (error*max_bps)
        # self.d_bps = min(max(min_bps, self.d_bps), max_bps)

        return d_euler_angles_left, d_euler_angles_right, self.bps
