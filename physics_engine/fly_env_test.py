import time

from fly_env import FlyEnv
from controller import HeightStabilizerController


if __name__ == '__main__':
    begin = time.time()

    controller = HeightStabilizerController(desired_height=0)
    env = FlyEnv(config_path='config.json', controller=controller)

    for i in range(20_000):
        if i % 10 == 0:
            print(f'Iteration {i}')
        is_done = env.step()
        if is_done:
            break

    print(f'Total time elapsed before rendering: {time.time() - begin}')

    env.render(x_axis=False, x_vs_z=True, render_3d=True, render_euler_angles=False, render_delta_euler_angles=False)
    env.close()

    end = time.time()
    print(f'Total time elapsed: {end - begin} s')
