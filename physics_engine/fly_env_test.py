import time
import numpy as np

from fly_env import FlyEnv
from controller import HeightStabilizerController, PitchStabilizerController


if __name__ == '__main__':
    begin = time.time()

    env = FlyEnv(config_path='./physics_engine/config.json')
    controller = HeightStabilizerController(desired_height=0, max_dev=4)
    # controller = PitchStabilizerController()
    action = np.zeros(env.action_space.shape)


    for i in range(20_000):
        if i % 10 == 0:
            print(f'Iteration {i}')

        obs, reward, is_done, info = env.step(action)
        if is_done:
            break
        action = controller.respond(obs)

    print(f'Total time elapsed before rendering: {time.time() - begin}')

    env.render(x_axis=False, x_vs_z=False, render_3d=False, render_3d_plotly=True, render_euler_angles=False,
               render_delta_euler_angles=False)
    env.close()
