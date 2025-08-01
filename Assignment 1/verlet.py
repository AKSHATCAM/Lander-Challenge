import numpy as np
import matplotlib.pyplot as plt

# Constants
m = 1
k = 1
x = 0
v = 1
t_max = 1000
dt = 1.00001
n_steps = int(t_max / dt)

# Verlet initial conditions
x_verlet = x
v_verlet = v

# Arrays for position and velocity
x_listverlet = [x_verlet]
v_listverlet = [v_verlet]

# Time array created up front
t_array = np.arange(0, t_max + dt, dt)

# First step (from Taylor expansion)
a = -k * x / m
x_next = x + v * dt + 0.5 * a * dt**2
x_listverlet.append(x_next)
v_listverlet.append(v_verlet)  # Optional: same initial velocity for consistency

# Verlet loop
for i in range(2, len(t_array)):
    a = -k * x_listverlet[-1] / m
    x_new = 2 * x_listverlet[-1] - x_listverlet[-2] + dt**2 * a
    x_listverlet.append(x_new)
    v_estimate = (x_listverlet[-1] - x_listverlet[-3]) / (2 * dt)
    v_listverlet.append(v_estimate)

# Convert to NumPy arrays
x_array = np.array(x_listverlet)
v_array = np.array(v_listverlet)

# Plotting
plt.figure(2)
plt.clf()
plt.title("Verlet Integration of Harmonic Oscillator")
plt.xlabel('Time (s)')
plt.grid()
plt.plot(t_array, x_array, label='x (m)')
plt.plot(t_array, v_array, label='v (m/s)')
plt.legend()
plt.show()
