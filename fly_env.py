from computations import AeroModel, fly_solve_diff, wing_angles, DEG2RAD, RAD2DEG
import utils
from flight_renderer_3d import save_3d_trajectory

import gym
from gym import spaces

import numpy as np
from scipy.integrate import solve_ivp
from scipy import interpolate

import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import json

class FlyEnvState:
    '''
    Stores information on the fly at any given point in the simulation
    '''

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
    '''Stores information on a wing at any given point in the simulation'''
    def __init__(self, psi, theta, phi):
        self.psi = psi
        self.theta = theta
        self.phi = phi

    def as_vec(self):
        '''Concatenates psi theta and phi vectors into one vector representation'''
        return np.concatenate([self.psi, self.theta, self.phi])


class FlyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config_path):
        # Define essential gym attributes
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(25,))
        self.action_space = spaces.Box(-1, 1, shape=(13,))

        # Save configuration path
        self.config_path = config_path

        # Initialize all variables and read config file:
        self.config = None
        self.__reset()

        self.MAX_ANGLE = 25  # TODO arbitrary constant

    def __load_config(self, config_path):
        if self.config is None:
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
        '''
        Initialize an AeroModel object for each wing. Called in self.reset().
        '''
        return AeroModel(self.config['wing']['span'], self.config['wing']['chord'], self.config['gen']['rho'],
                         self.config['aero']['r22'], self.config['aero']['CLmax'], self.config['aero']['CDmax'],
                         self.config['aero']['CD0'], self.config['wing']['hingeL'], self.config['wing']['ACloc']), \
            AeroModel(self.config['wing']['span'], self.config['wing']['chord'], self.config['gen']['rho'],
                      self.config['aero']['r22'], self.config['aero']['CLmax'], self.config['aero']['CDmax'],
                      self.config['aero']['CD0'], self.config['wing']['hingeR'], self.config['wing']['ACloc'])

    def __change_wing_phi(self, action):
        '''
        This method is copied from the matlab version of the simulation.
        action: ndarray of shape (2,) -- change in left and right wings
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
        '''
        Initialize simulation's time vectors. This is pretty much copied from the matlab version of the simulation.
        See matlab version for reference.
        '''

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

    def __reset(self):
        '''
        Loads the configuration given in the constructor, and initializes all state variables of the simulation
        '''

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
        self.y_axis = [np.zeros(3)]
        self.z_axis = [np.zeros(3)]
        self.bps_history = [self.wing_freq / (2 * np.pi)]

        self.euler_angles_history = [np.zeros(3)]
        self.delta_euler_angles_history = []

        # Times
        self.__create_time_vectors()

        return self.__get_current_state()

    def reset(self):
        # Make separate private method since we're using it in the constructor
        return self.__reset()
    
    def step(self, action):
        '''
        Execute one step in the simulation. This updates the fly's position by solving ODEs.
        action: a vector containing various parameters to be adjusted, see Controller class
        for documentation
        '''
        
        if self.__is_done():
            return  # done

        # Record history for plotting later
        self.bps_history.append(self.wing_freq / (2 * np.pi))
        self.delta_euler_angles_history.append(np.array([0, action[2], 0]))

        # Apply action
        d_phi_left, d_phi_right = action[2], action[5] # See Controller documentation
        d_bps = action[6]

        self.__change_wing_phi((d_phi_left, d_phi_right))
        self.wing_freq = (self.wing_freq / (2 * np.pi) + d_bps) * 2 * np.pi

        # Compute time step information needed for solve_ivp
        step_time_start = self.tvec[self.decision_point_idx[self.time_step]]
        step_time_end = self.tvec[self.decision_point_idx[self.time_step + 1]]
        time_range = (step_time_start, step_time_end)
        delta_t = time_range[1] - time_range[0]

        # Compute next state by solving ODEs
        func = lambda t, y, fly_env: fly_solve_diff(t, y, fly_env)[0]
        ode_sol = solve_ivp(func, time_range, self.__get_body_state_vec(),
                            method=self.config['solver']['method'],
                            args=[self],
                            atol=self.config['solver']['atol'], rtol=self.config['solver']['rtol'],
                            max_step=self.config['gen']['MaxStepSize'])
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

        # Compute Xax and Zax from Roni's matlab code
        self.x_axis.append(rotation_matrix @ np.array([1, 0, 0]))
        self.y_axis.append(rotation_matrix @ np.array([0, 1, 0]))
        self.z_axis.append(rotation_matrix @ np.array([0, 0, 1]))

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
        measured_state.euler_angles[0] = utils.standardize_angle(measured_euler_angles[0]) # TODO: vectorize these 3 lines
        measured_state.euler_angles[1] = utils.standardize_angle(measured_euler_angles[1])
        measured_state.euler_angles[2] = utils.standardize_angle(measured_euler_angles[2])
        self.euler_angles_history.append(self.curr_euler_angles)

        # Increment time step
        self.time_step += 1

        return measured_state, 0, self.__is_done(), None

    def render(self, mode='human', x_axis=False, y_axis=False, z_axis=False, render_3d=False, render_3d_plotly=False,
               x_vs_z=False, render_euler_angles=False, render_delta_euler_angles=False):
        if not self.__is_done():
            return Exception("Cannot render since simulation hasn't ended")

        self.euler_angles_history = np.array(self.euler_angles_history) * RAD2DEG
        self.delta_euler_angles_history = np.array(self.delta_euler_angles_history) * RAD2DEG

        # pitch_history = [utils.standardize_angle(x * DEG2RAD) * RAD2DEG for x in self.euler_angles_history[:, 1]]
        pitch_history = np.vectorize(utils.standardize_angle)(self.euler_angles_history[:, 1] * DEG2RAD) * RAD2DEG
        
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

        if render_3d_plotly:
            com, x_axis, y_axis = np.array(self.locs), np.array(self.x_axis), np.array(self.y_axis)
            times = self.tvec[self.decision_point_idx[:self.time_step + 1]]
            times = np.array(times).reshape(-1, 1) * 1000
            # bps = np.array(self.bps_history).reshape(-1, 1)
            flight_data = np.concatenate([com, x_axis, y_axis, times], axis=1)
            # plot_3d_trajectory_in_browser(flight_data, color_property='t_ms')
            save_3d_trajectory(flight_data, path='plotly_flight', color_property='t_ms')
        
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
            plt.plot(pitch_history, label='pitch')
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

