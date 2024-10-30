'''
This script trains a Q-Table using Q-Learning with the epsilon-greedy policy.
How to run: "python3 <session name> <config file> <q-table path>"
q-table path is optional
'''

import os
import shutil
import sys
import numpy as np
import random

from DiscreteRLFlyEnv import DiscreteFlyEnv
from computations import DEG2RAD, RAD2DEG

import datetime
import pickle
import concurrent.futures
import time


def log(s):
    '''Helper function for writing to the script's log file. Records date.'''

    log_file_handle.write(f'[{str(datetime.datetime.now())}]: {s}\n')
    

def save_array(title, a):
    '''Helper function for saving an array to file using Pickle.'''

    with open(title, 'wb') as f:
        pickle.dump(a, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_array(title, a):
    '''Helper function for loading an array from a file using Pickle.'''

    with open(title, 'rb') as f:
        return pickle.load(f)
    

def create_env():
    '''Create new environment object from configuration file'''
    # TODO: add these parameters to the config file
    state_space = (59, 9) # These are the dimensions we chose in the meantime
    n_actions = 7 # These are the dimensions we chose in the meantime
    env = DiscreteFlyEnv(config_path='config.json', state_space=state_space, action_space=n_actions)
    return env


def epsilon_greedy_policy(state, epsilon, Qtable):
    '''Standard epsilon-greedy policy to balance exploration and exploitation of the Q-Table.'''

    random_int = random.uniform(0, 1)
    if random_int > epsilon:
        action = np.argmax(Qtable[tuple(state)])
    else:
        action = env.action_space.sample()
    return action


def train(n_level, n_training_episodes, decay_rate, env, max_steps, Qtable, random_pitch=False, random_pitch_dot=False):
    '''
    Trains the Q-Table for the given number of episodes using the epsilon-greedy policy.
    Each episode is given a maximum of 30 seconds.
    The Q-Table, rewards history, pitch history, and state heatmap are recorded and saved intermittently.
    '''

    def episode_code():
        log(f'Level {n_level + 1}/{n_levels}, episode: {episode + 1}/{n_training_episodes}')

        episode_pitch_history = []

        # Periodic save
        if episode % 10 == 0:
            log('Periodic save...')
            log_file_handle.flush()
            # os.fsync()
            np.save('q_table_iter.npy', Qtable)
            save_array('rewards_iter.pickle', rewards)
            save_array('pitch_history_iter.pickle', pitch_history)
            np.save('state_heatmap_iter.npy', state_heatmap)

        # Reset the environment
        state = env.reset()

        # Start with random pitch
        if random_pitch:
            random_start_pitch = random.uniform(-np.pi, np.pi)
            env.curr_euler_angles[1] = random_start_pitch
            log(f'Random start pitch = True, pitch={random_start_pitch}')
        if random_pitch_dot:
            random_start_delta_pitch = random.uniform(-40, 40)
            env.curr_angular_vel[1] = random_start_delta_pitch
            log(f'Random start pitch dot = True, pitch dot={random_start_delta_pitch}')

        for step in range(max_steps):
            # Reduce epsilon, exponential decay rate
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * step)

            action = epsilon_greedy_policy(state, epsilon, Qtable)

            new_state, reward, done, info = env.step(action)

            # Record
            rewards.append(reward)
            state_heatmap[tuple(state)] += 1
            episode_pitch_history.append(env.pitch * RAD2DEG)

            # Update table
            Qtable[tuple(state)][action] = Qtable[tuple(state)][action] + learning_rate * (
                    reward + gamma * np.max(Qtable[tuple(new_state)]) - Qtable[tuple(state)][action])

            # If done, finish the episode
            if done:
                log(f'Finished episode {episode} after {step}/{max_steps} steps')
                break

            state = new_state  # Update state
        # Stats
        pitch_history.append(episode_pitch_history)

    # Running the episode code with a time limit
    time_limit = 30
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for episode in range(n_training_episodes):
            future = executor.submit(episode_code)
            try:
                log('Starting future')
                begin_time = time.time()
                # concurrent.futures.wait(future, timeout=time_limit)
                future.result(timeout=time_limit)
                log(f'Future finished. Duration={time.time() - begin_time:.2f}s')
            except concurrent.futures.TimeoutError:
                log(f'Execution exceeded the time limit of {time_limit} seconds and was terminated.')
            except Exception as e:
                log(f'[EMERGENCY] An runtime error has occurred: {str(e)}. Attempting to reconstruct env and continue to next episode.')
                env = create_env() # Reconstruct env
                
    return Qtable


def curriculum_train(Qtable, n_levels, level_durations, max_steps, decay_rates):
    '''
    Runs the training procedure with the given curriculum.
    '''

    log('Beginning curriculum train')
    
    random_pitch = False
    random_pitch_dot = False
    for level in range(n_levels):
        log(f'Starting level {level + 1}/{n_levels}')
        if level > 6:
            random_pitch = True
            random_pitch_dot = False
        if level > 9:
            random_pitch = False
            random_pitch_dot = True
        if level > 12:
            random_pitch = True
            random_pitch_dot = True

        Qtable = train(level, level_durations[level], decay_rates[level], env, max_steps[level], q_table, random_pitch, random_pitch_dot)

    return Qtable



# How to run: "python3 <session name> <config file> <q-table path>"
# q-table path is optional
# TODO: use argeparse to handle command line arguments
if __name__ == '__main__':
    # Process program arguments
    if len(sys.argv[1]) <= 2 or len(sys.argv) > 4:
        print('Invalid number of program arguments')
        exit()

    # Create session folder with the given name. All files created will reside in it.
    session_name = sys.argv[1]
    session_folder = f'/cs/labs/tsevi/dink/Artificial-Fly/physics_engine/{session_name}'
    os.makedirs(session_folder, exist_ok=True)
    # Copy config file to session directory
    config_file = sys.argv[2]
    env = create_env() # Initialize env
    shutil.copyfile(config_file, f'{session_folder}/config.json')
    # Move into session folder
    os.chdir(session_folder)

    # Create log file
    LOG_FILE = f'{session_name}_log.txt'
    log_file_handle = open(LOG_FILE, 'a+')
    log(f'------------------------------\nBeginning session {session_name}, creating session folder at {session_folder}...')
    
    # Announce start:
    log(f'----------TRAIN START----------')

    # If given a q table, load it, otherwise initialize from a standard gaussian.
    q_table_path = f'../{sys.argv[3]}' if len(sys.argv) > 3 else None
    if q_table_path is not None:
        log(f'Loading q_table from file: {q_table_path}')
        q_table = np.load(q_table_path)
    else:
        log(f'Initializing new q_table (Gaussian)')
        # q_table = np.zeros((*env.observation_space.nvec, env.action_space.n))
        q_table = np.random.standard_normal((*env.observation_space.nvec, env.action_space.n))
    
    # Fixed Learning Parameters
    learning_rate = 0.2
    gamma = 0.7
    max_epsilon = 1.0
    min_epsilon = 0.05

    log(f'Initializing hyperparameters: lr={learning_rate}, gamma={gamma}, min_epsilon={min_epsilon}, max_epsilon={max_epsilon}')

    # Training stats
    if os.path.exists('rewards_history_iter.pickle'):
        log('Loading existing rewards history')
        with open('rewards_iter.pickle', 'rb') as f:
            rewards = pickle.load(f)
    else:
        log('Initializing new rewards history')
        rewards = []

    if os.path.exists('pitch_history_iter.pickle'):
        log('Loading existing pitch history')
        with open('pitch_history_iter.pickle', 'rb') as f:
            pitch_history = pickle.load(f)
    else:
        log('Initializing new pitch history')
        pitch_history = []

    if os.path.exists('state_heatmap_iter.npy'):
        log('Loading existing state heatmap')
        state_heatmap = np.load('state_heatmap_iter.npy')
    else:
        log('Initializing new state heatmap')
        state_heatmap = np.zeros((env.observation_space.nvec))

    
    # STAGE 1: basic training
    
    max_steps = [10, 15, 20, 30, 40, 50, 80]
    decay_rates = [0.3, 0.3, 0.2, 0.15, 0.12, 0.08, 0.07]
    level_durations = [300, 300, 300, 300, 300, 400, 600]
    
    # STAGE 2: random pitch initializations
    
    max_steps += [50, 60, 70]
    decay_rates += [0.2, 0.15, 0.1]
    level_durations += [300, 300, 500]
    
    # STAGE 3: random pitch dot initializations
    
    max_steps += [50, 60, 70]
    decay_rates += [0.2, 0.15, 0.1]
    level_durations += [300, 300, 500]
    
    # STAGE 4: random pitch and pitch dot initializations
    
    max_steps += [100, 100, 100]
    decay_rates += [0.07, 0.07, 0.09]
    level_durations += [300, 400, 500]
    n_levels = len(max_steps)

    # TRAIN
    q_table = curriculum_train(q_table, n_levels, level_durations, max_steps, decay_rates)

    # Save q table
    np.save('q_table_final.npy', q_table)
    save_array('rewards_final.pickle', rewards)
    save_array('pitch_history_final.pickle', pitch_history)
    np.save('state_heatmap_final.npy', state_heatmap)
    
    log(f'----------TRAIN END----------')
