from Functions import euler_gravity, verlet_gravity
import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
mass_lander = 1.0         # Mass of the lander (kg)
mass_planet = 6.42e23     # Mass of Mars (kg)
rad_planet = 3.39e6       # Radius of Mars (m)
# Initial conditions (3D)
position = np.array([0, 10e6, 0])  # Start 10,000 km from planet center
velocity = np.array([0, 0, 0])   # Initial tangential velocity (m/s)

# Time setup
t_max = 1000  # Total simulation time (s)
dt = 1        # Timestep (s)

# Run simulation
t_array, x_array, v_array = verlet_gravity(position, velocity, mass_lander, mass_planet, dt, t_max)
t_array_euler, x_array_euler, v_array_euler = euler_gravity(position, velocity, mass_lander, mass_planet, dt, t_max)
mars_radius = 3.39e6  # m

altitude_verlet = np.linalg.norm(x_array, axis=1) - mars_radius
altitude_verlet[altitude_verlet < 0] = 0  # Set altitude to 0 if below surface

altitude_euler = np.linalg.norm(x_array_euler, axis=1) - mars_radius
altitude_euler[altitude_euler < 0] = 0  # Set altitude to 0 if below surface

plt.figure(figsize=(10, 5))
plt.title("Altitude vs Time (Scenario 1: Free Fall), Verlet Integration")
plt.plot(t_array, altitude_verlet, label='Altitude (m)', color='blue')
plt.xlabel("Time (s)")
plt.ylabel("Altitude above Mars (m)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
plt.title("Altitude vs Time (Scenario 1: Free Fall), Euler Integration")
plt.plot(t_array_euler, altitude_euler, label='Altitude (m)', color='blue')
plt.xlabel("Time (s)")
plt.ylabel("Altitude above Mars (m)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()