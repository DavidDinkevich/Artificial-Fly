from computations import AeroModel, fly_solve_diff, wing_angles, DEG2RAD, RAD2DEG
import utils

import gym
from gym import spaces

# Numpy/Scipy imports
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy import interpolate

import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import json


class FlyEnvState:
    def __init__(self):
        self.loc = None
        self.body_vel = None
        self.body_angular_vel = None
        self.euler_angles = None
        self.bps = None
        self.wing_state_left = None
        self.wing_state_right = None

    def __str__(self):
        return '\t'.join(f"{key}={value}" for key, value in vars(self).items())


class WingState:
    def __init__(self, psi, theta, phi):
        self.psi = psi
        self.theta = theta
        self.phi = phi


class FlyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config_path, controller: 'Controller' = None):
        # Check if we're given a controller
        self.controller = controller

        # Define essential gym attributes
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(25,))
        self.action_space = spaces.Box(-1, 1, shape=(13,))

        # Save configuration path
        self.config_path = config_path

        # Initialize all variables and read config file:
        self.reset(controller=controller)

        self.MAX_ANGLE = 25  # TODO const

    def __load_config(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Load config json and convert all arrays to numpy arrays
        utils.convert_arrays_to_numpy_in_dict(self.config)

        # Convert angles from degrees to radians
        self.config['gen']['strkplnAng'] *= DEG2RAD
        self.config['gen']['I'] *= 1e-12
        for k, v in self.config['wing'].items():
            if k in ['psi', 'theta', 'phi', 'delta_psi', 'delta_theta']:
                self.config['wing'][k] = v * DEG2RAD
        for k, v in self.config['body'].items():
            self.config['body'][k] = v * DEG2RAD

        # Add misc. stuff
        self.wing_freq = self.config['wing']['bps'] * 2 * np.pi
        self.wing_beat_time = (2 * np.pi) / self.wing_freq
        self.step_time = self.config['gen']['MaxStepSize']

    def __create_wing_models(self):
        return AeroModel(self.config['wing']['span'], self.config['wing']['chord'], self.config['gen']['rho'],
                         self.config['aero']['r22'], self.config['aero']['CLmax'], self.config['aero']['CDmax'],
                         self.config['aero']['CD0'], self.config['wing']['hingeL'], self.config['wing']['ACloc']), \
            AeroModel(self.config['wing']['span'], self.config['wing']['chord'], self.config['gen']['rho'],
                      self.config['aero']['r22'], self.config['aero']['CLmax'], self.config['aero']['CDmax'],
                      self.config['aero']['CD0'], self.config['wing']['hingeR'], self.config['wing']['ACloc'])

    def __change_wing_phi(self, action):
        '''
        Applies the action to the wings and makes sure that:
         (1) The action values are within the allowed range of TODO
         (2) The wings do not exceed the allowed range of TODO
        :param action: ndarray of shape (2,) -- change in left and right wings
        :return: None
        '''
        delta_phi_l = action[0]
        delta_phi_r = action[1]

        # Change in middle point of the stroke (Left)
        self.curr_wing_state_left.phi[0] = self.default_wing_state_left.phi[0] - delta_phi_l / 2
        # Change in the amplitude of the stroke (Left)
        self.curr_wing_state_left.phi[1] = self.default_wing_state_left.phi[1] + delta_phi_l / 2
        # Change in middle point of the stroke (Right)
        self.curr_wing_state_right.phi[0] = self.default_wing_state_right.phi[0] + delta_phi_r / 2
        # Change in the amplitude of the stroke (Right)
        self.curr_wing_state_right.phi[1] = self.default_wing_state_right.phi[1] - delta_phi_r / 2

    def __create_time_vectors(self):
        sim_start_time, sim_end_time = self.config['gen']['tsim_in'], self.config['gen']['tsim_fin']

        # Time range [sim_start_time, sim_end_time] (inclusive)
        tmes = np.arange(sim_start_time, sim_end_time + self.wing_beat_time / 4, self.wing_beat_time / 2) # [0, 0.5T, T, 1.5T, ...]
        tmesQT = np.arange(self.wing_beat_time * 0.25, sim_end_time, self.wing_beat_time / 2) # [0.25T, 0.75T, 1.25T, ...]
        # Calculate all angles at the time vector. Then, find only the ones that are min phi (when the wing
        # is in its back stroke), they are the decision points
        anglesD = np.array([wing_angles(self.default_wing_state_left.psi, self.default_wing_state_left.theta,
                                        self.default_wing_state_left.phi, self.wing_freq,
                                        self.config['wing']['delta_psi'],
                                        self.config['wing']['delta_theta'], self.config['wing']['C'],
                                        self.config['wing']['K'],
                                        t)[0]
                            for t in tmesQT])
        indices_of_min_phi = np.where(anglesD[:, 2] > self.default_wing_state_left.phi[0])
        tdes = tmesQT[indices_of_min_phi]

        # Define the time vectors relevant to the user simulation time input
        self.tmes_sim = tmes[np.where(tmes <= sim_end_time)]
        self.tdes_sim = tdes[np.where(tdes <= sim_end_time)]
        # Find the first decision point (that have 2 measure points before it)
        val = np.unique(np.concatenate((self.tdes_sim, self.tmes_sim)))
        first_decision_pt_idx = np.where(self.tdes_sim[0] == val)[0][0]
        # Check if the fly has 2 measurement points before the first decision point. Otherwise, take the
        # next decision point to be the first
        if first_decision_pt_idx < 2:
            # Take the second decision point
            self.tdes_sim = self.tdes_sim[1:]
            FidtSecmeas = val[first_decision_pt_idx + 1:first_decision_pt_idx + 3]
            self.tmes_sim = self.tmes_sim[np.where(self.tmes_sim >= FidtSecmeas[0])[0]]

        self.simtime_tvec = np.arange(sim_start_time, sim_end_time + self.step_time / 2, self.step_time)
        self.tvec = np.unique(
            np.concatenate((self.tdes_sim, self.simtime_tvec)))  # TODO: comes out different than matlab
        self.decision_point_idx = np.concatenate(([0], np.nonzero(np.isin(self.tvec, self.tdes_sim))[0]))

    def __get_body_state_vec(self):
        return np.concatenate((self.curr_body_vel, self.curr_angular_vel, self.curr_euler_angles))

    def __get_current_state(self):
        state = FlyEnvState()
        state.loc = self.locs[-1]
        state.body_vel = self.curr_body_vel
        state.body_angular_vel = self.curr_angular_vel
        state.euler_angles = self.curr_euler_angles
        state.bps = self.wing_freq / (2 * np.pi)
        state.wing_state_left = self.curr_wing_state_left
        state.wing_state_right = self.curr_wing_state_right
        return state

    def __is_done(self):
        return self.time_step == len(self.decision_point_idx) - 1

    def reset(self, controller: 'Controller' = False):
        # Initialize time settings
        self.time_step = 0

        # Load configuration file
        self.__load_config(self.config_path)

        # Initialize wing models
        self.wing_model_left, self.wing_model_right = self.__create_wing_models()

        # Create initial state variables
        self.curr_body_vel = self.config['body']['BodIniVel']
        self.curr_angular_vel = self.config['body']['BodInipqr']
        self.curr_euler_angles = self.config['body']['BodIniang']

        # Save initial wing states
        self.default_wing_state_left = WingState(self.config['wing']['psi'][:2].copy(),
                                                 self.config['wing']['theta'][:2].copy(),
                                                 self.config['wing']['phi'][:2].copy())
        self.default_wing_state_right = WingState(self.config['wing']['psi'][2:].copy(),
                                                  self.config['wing']['theta'][2:].copy(),
                                                  self.config['wing']['phi'][2:].copy())
        # Create variables to store current wing states
        self.curr_wing_state_left = WingState(self.config['wing']['psi'][:2].copy(),
                                              self.config['wing']['theta'][:2].copy(),
                                              self.config['wing']['phi'][:2].copy())
        self.curr_wing_state_right = WingState(self.config['wing']['psi'][2:].copy(),
                                               self.config['wing']['theta'][2:].copy(),
                                               self.config['wing']['phi'][2:].copy())

        # Initial fly location
        self.vlabs = [np.zeros(3)]
        self.locs = [np.zeros(3)]
        self.x_axis = [np.zeros(3)]
        self.z_axis = [np.zeros(3)]

        self.euler_angles_history = []
        self.delta_euler_angles_history = []

        # Times
        self.__create_time_vectors()

        # Controller
        self.controller = controller

    def step(self, action=(0, 0)):
        if self.__is_done():
            return  # done

        step_time_start = self.tvec[self.decision_point_idx[self.time_step]]
        step_time_end = self.tvec[self.decision_point_idx[self.time_step + 1]]
        time_range = (step_time_start, step_time_end)
        delta_t = time_range[1] - time_range[0]

        # Compute next state by solving ODEs
        # time_vec = np.linspace(time_range[0], time_range[-1], 100)
        func = lambda t, y, fly_env: fly_solve_diff(t, y, fly_env)[0]
        ode_sol = solve_ivp(func, time_range, self.__get_body_state_vec(),
                            method=self.config['solver']['method'], t_eval=None,
                            args=[self],
                            atol=self.config['solver']['atol'], rtol=self.config['solver']['rtol'])
        # Copy solution into state (this is the new state)
        self.curr_body_vel = ode_sol.y[:3, -1]
        self.curr_angular_vel = ode_sol.y[3:6, -1]
        self.curr_euler_angles = ode_sol.y[6:, -1]

        # Transform body-frame velocity to lab-frame and integrate to get position
        _, _, wingout_r = fly_solve_diff(step_time_end, self.__get_body_state_vec(), self)
        # Compute Vlab
        rotation_matrix = wingout_r[2]
        Vlab = rotation_matrix @ self.curr_body_vel
        self.vlabs.append(Vlab)

        # Use trapezoidal rule to incrementally update the position
        new_loc = self.locs[-1] + 0.5 * delta_t * (self.vlabs[-1] + self.vlabs[-2])
        self.locs.append(new_loc)

        # Compute Xax and Zax from Roni's code
        self.x_axis.append(rotation_matrix @ np.array([1, 0, 0]))
        self.z_axis.append(rotation_matrix @ np.array([0, 0, 1]))

        if self.controller:
            # Compute measurement points
            # Interpolate the solution at measurement points
            interp_func = interpolate.interp1d(ode_sol.t, ode_sol.y, axis=1)
            measure_points = interp_func(self.tmes_sim[2 * self.time_step:2 * (self.time_step + 1)])
            measured_uvw = np.mean(measure_points[0:3, :], axis=1)
            measured_pqr = np.mean(measure_points[3:6, :], axis=1)
            measured_euler_angles = np.mean(measure_points[6:, :], axis=1)
            # Final measured state at decision point
            measured_state = self.__get_current_state()
            measured_state.body_vel = measured_uvw
            measured_state.body_angular_vel = measured_pqr
            measured_state.euler_angles = measured_euler_angles

            d_euler_angles_left, d_euler_angles_right, new_bps = self.controller.respond(measured_state)

            # print(new_bps)
            self.wing_freq = new_bps * (2 * np.pi)
            self.__change_wing_phi((d_euler_angles_left[2], d_euler_angles_right[2]))

            self.euler_angles_history.append(self.curr_euler_angles)
            self.delta_euler_angles_history.append(d_euler_angles_left)

        # Increment time step
        self.time_step += 1

        return self.__is_done()

    def render(self, mode='human', x_axis=False, y_axis=False, z_axis=False, render_3d=False, x_vs_z=False,
               render_euler_angles=False, render_delta_euler_angles=False):
        if not self.__is_done():
            return Exception("Cannot render since simulation hasn't ended")

        self.euler_angles_history = np.array(self.euler_angles_history) * RAD2DEG
        self.delta_euler_angles_history = np.array(self.delta_euler_angles_history) * RAD2DEG
        self.vlabs = np.array(self.vlabs)

        locs_meters = np.array(self.locs) * 1000

        x_values = locs_meters[:, 0]
        y_values = locs_meters[:, 1]
        z_values = locs_meters[:, 2]

        x_head = locs_meters + self.x_axis

        # Plot x values per time
        if x_axis:
            plt.figure(figsize=(8, 6))
            plt.plot(x_values, label='X values')
            plt.plot(x_head[:, 0], label='X head')
            plt.xlabel('Time')
            plt.ylabel('X coordinate')
            plt.title('X values per time')
            plt.legend()
            plt.show()

        # Plot y values per time
        if y_axis:
            plt.figure(figsize=(8, 6))
            plt.plot(y_values, label='Y values')
            plt.xlabel('Time')
            plt.ylabel('Y coordinate')
            plt.title('Y values per time')
            plt.legend()
            plt.show()

        # Plot z values per time
        if z_axis:
            plt.figure(figsize=(8, 6))
            plt.plot(z_values, label='Z values')
            plt.xlabel('Time')
            plt.ylabel('Z coordinate')
            plt.title('Z values per time')
            plt.legend()
            plt.show()

        # Plot 3D points per time
        if render_3d:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x_values, y_values, z_values, c='b', marker='o')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('3D Points per Time')
            plt.show()

        if x_vs_z:
            # Plot x_values against z_values
            plt.figure(figsize=(8, 6))
            plt.plot(x_values, z_values, label='z values')
            plt.plot(x_head[:, 0], x_head[:, 2], label='head')
            plt.xlabel('X values')
            plt.ylabel('Z values')
            plt.title('Plot of X values against Z values')
            plt.legend()
            plt.grid(True)
            plt.show()

        # Plot pitch over time
        if render_euler_angles:
            plt.figure(figsize=(8, 6))
            plt.plot(self.euler_angles_history[:, 0], label='roll')
            plt.plot(self.euler_angles_history[:, 1], label='pitch')
            plt.plot(self.euler_angles_history[:, 2], label='yaw')
            plt.xlabel('Time (ms)')
            plt.ylabel('Degrees')
            plt.title('Euler Angles over Time (deg/ms)')
            plt.legend()
            plt.show()
        if render_delta_euler_angles:
            plt.figure(figsize=(8, 6))
            plt.plot(self.delta_euler_angles_history[:, 0], label='roll')
            plt.plot(self.delta_euler_angles_history[:, 1], label='pitch')
            plt.plot(self.delta_euler_angles_history[:, 2], label='yaw')
            plt.xlabel('Time (ms)')
            plt.ylabel('Degrees')
            plt.title('Change in Euler Angles over Time (deg/ms)')
            plt.legend()
            plt.show()

    def close(self):
        pass

