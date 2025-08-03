import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
mass_lander = 1.0         # Mass of the lander (kg)
mass_planet = 6.42e23     # Mass of Mars (kg)

# Initial conditions (3D)
position = np.array([0, 10e6, 0])  # Start 3400 km from planet center
velocity = np.array([0, 0, 0])   # Initial tangential velocity (m/s)

# Time setup
t_max = 1000  # Total simulation time (s)
dt = 1        # Timestep (s)

# Gravitational force function
def gravity_force(mass_planet, mass_lander, position):
    r = np.linalg.norm(position)
    if r == 0:
        return np.zeros(3)  # Avoid division by zero
    force_mag = -G * mass_planet * mass_lander / r**2
    return force_mag * position / r

# Euler method for comparison (optional)
def euler_gravity(position, velocity, mass_lander, mass_planet, dt, t_max):
    t_array = np.arange(0, t_max, dt)
    pos_list = []
    vel_list = []

    position = position.copy()
    velocity = velocity.copy()

    for _ in t_array:
        pos_list.append(position.copy())
        vel_list.append(velocity.copy())

        a = gravity_force(mass_planet, mass_lander, position) / mass_lander
        position += dt * velocity
        velocity += dt * a

    return np.array(pos_list), np.array(vel_list)

# Verlet method
def verlet_gravity(position, velocity, mass_lander, mass_planet, dt, t_max):
    t_array = np.arange(0, t_max + dt, dt)
    n = len(t_array)

    x_array = np.zeros((n, 3))
    v_array = np.zeros((n, 3))

    x = position.copy()
    v = velocity.copy()
    x_array[0] = x
    v_array[0] = v

    # First step using Taylor expansion
    a0 = gravity_force(mass_planet, mass_lander, x) / mass_lander
    x_next = x + v * dt + 0.5 * a0 * dt**2
    x_array[1] = x_next

    # Verlet loop
    for i in range(2, n):
        a = gravity_force(mass_planet, mass_lander, x_array[i - 1]) / mass_lander
        x_array[i] = 2 * x_array[i - 1] - x_array[i - 2] + dt**2 * a
        v_array[i - 1] = (x_array[i] - x_array[i - 2]) / (2 * dt)

    # Estimate final velocity
    v_array[-1] = (x_array[-1] - x_array[-3]) / (2 * dt)

    return t_array, x_array, v_array

# Run simulation
t_array, x_array, v_array = verlet_gravity(position, velocity, mass_lander, mass_planet, dt, t_max)

mars_radius = 3.39e6  # m
altitude = np.linalg.norm(x_array, axis=1) - mars_radius

plt.figure(figsize=(10, 5))
plt.title("Altitude vs Time (Scenario 1: Free Fall)")
plt.plot(t_array, altitude, label='Altitude (m)', color='blue')
plt.xlabel("Time (s)")
plt.ylabel("Altitude above Mars (m)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
