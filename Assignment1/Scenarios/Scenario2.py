from Functions import euler_gravity, verlet_gravity
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
G = 6.67430e-11   # Gravitational constant (m^3 kg^-1 s^-2)
mass_lander = 1.0 # Mass of the lander (kg)
mass_planet = 6.42e23  # Mass of Mars (kg)
rad_planet = 3.39e6    # Radius of Mars (m)

# Initial conditions (3D)
radius = 8e6
initial_velocity = np.sqrt(G * mass_planet / radius)
position = np.array([0, radius, 0])         # Start position
velocity = np.array([initial_velocity, 0, 0])  # Tangential velocity

# Time setup
t_max = 1000  # Total simulation time (s)
dt = 1        # Timestep (s)

# Run simulation
t_array, x_array, v_array = verlet_gravity(position, velocity, mass_lander, mass_planet, dt, t_max)
t_array_euler, x_array_euler, v_array_euler = euler_gravity(position, velocity, mass_lander, mass_planet, dt, t_max)

# Extract coordinates
X, Y, Z = x_array[:, 0], x_array[:, 1], x_array[:, 2]
X_e, Y_e, Z_e = x_array_euler[:, 0], x_array_euler[:, 1], x_array_euler[:, 2]

# Plotting
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

ax.plot(X, Y, Z, color='royalblue', linewidth=2, label='Verlet Trajectory')
ax.plot(X_e, Y_e, Z_e, color='orangered', linestyle='--', linewidth=1.5, label='Euler Trajectory')

# Start and end markers
ax.scatter(X[0], Y[0], Z[0], color='green', marker='o', s=60, label='Start')
ax.scatter(X[-1], Y[-1], Z[-1], color='black', marker='x', s=60, label='End')

# Labels and title
ax.set_title('Lander Trajectory Simulation', fontsize=14, fontweight='bold')
ax.set_xlabel('X (m)', fontsize=12)
ax.set_ylabel('Y (m)', fontsize=12)
ax.set_zlabel('Z (m)', fontsize=12)
ax.legend()

# Equal aspect ratio
ax.set_box_aspect([1, 1, 1])

plt.tight_layout()
plt.show()
