# Perturbation Rejection - Fruit Fly Stabilization Simulation

This project builds on the work of Roni Amit, who built a simulation that models a fruit fly’s movement over time. The simulation was originally written in MATLAB, and the first step I took was to convert it to Python due to my intention to use helpful Python machine learning libraries.
The main goal of the project was to train a model that has control over the fruit fly’s wings to achieve perturbation rejection. In other words, the model would stabilize the fly so that it can hover in place, and if it is perturbed (e.g., a torque force is applied to it), the model would be capable of restabilizing the fly as quickly as possible.
Stabilizing the fly involves stabilizing its body on all three axes; for simplicity’s sake, we began by focusing only on pitch. We successfully trained a model that achieves stabilization by changing the front stroke angle of each wing as a function of the fly’s current pitch and its angular velocity.
<br><br>
Example of fly stabilizing:
<br><br>
![Example of fly stabilizing](https://github.com/daviddinkevich/Artificial-Fly/blob/main/trajectory_image.png?raw=true)
