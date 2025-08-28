from Functions import euler_gravity, verlet_gravity
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def elliptical_orbit_initial_vel(initial_position, mass_planet, eccentricity):
    # Periapsis distance
    r_p = np.linalg.norm(initial_position)
    
    # Semi-major axis (a) and semi-minor axis (b)
    a = r_p / (1 - eccentricity)
    b = a * np.sqrt(1 - eccentricity**2)

    # Gravitational parameter
    mu = G * mass_planet
    
    # Velocity at periapsis from vis-viva equation
    v = np.sqrt(mu * (2 / r_p - 1 / a))
    return v

# Constants
G = 6.67430e-11   # Gravitational constant (m^3 kg^-1 s^-2)
mass_lander = 1.0 # Mass of the lander (kg)
mass_planet = 6.42e23  # Mass of Mars (kg)
rad_planet = 3.39e6    # Radius of Mars (m)

# Initial conditions (3D)
eccentricity = 0.5   # Must be between 0 and 1 for ellipse
radius = 8e6
position = np.array([0, radius, 0])  # Periapsis along Y-axis
initial_velocity = elliptical_orbit_initial_vel(position, mass_planet, eccentricity)
velocity = np.array([initial_velocity, 0, 0])  # Tangential direction

# Time setup
t_max = 20000  # Longer sim for elliptical orbit
dt = 10        # Timestep (s)

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
ax.set_title('Elliptical Orbit Simulation', fontsize=14, fontweight='bold')
ax.set_xlabel('X (m)', fontsize=12)
ax.set_ylabel('Y (m)', fontsize=12)
ax.set_zlabel('Z (m)', fontsize=12)
ax.legend()

# Equal aspect ratio
ax.set_box_aspect([1, 1, 1])

plt.tight_layout()
plt.show()
