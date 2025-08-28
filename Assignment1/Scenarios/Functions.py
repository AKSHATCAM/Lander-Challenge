import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
mass_lander = 1.0         # Mass of the lander (kg)
mass_planet = 6.42e23     # Mass of Mars (kg)

# Initial conditions (3D)
position = np.array([0, 10e6, 0])  # Start 10,000 km from planet center
velocity = np.array([0, 0, 0])   # Initial tangential velocity (m/s)

# Time setup
t_max = 1000  # Total simulation time (s)
dt = 1        # Timestep (s)

def gravity_force(mass_planet, mass_lander, position):
    # Universal gravitational constant
    G = 6.67430e-11  
    r = np.linalg.norm(position)
    if r == 0:
        return np.zeros(3)  # avoid division by zero
    # Force vector on lander
    return -G * mass_planet * mass_lander * position / r**3

def euler_gravity(position, velocity, mass_lander, mass_planet, dt, t_max):
    t_array = np.arange(0, t_max + dt, dt)
    n = len(t_array)

    vel = np.zeros((n, 3))
    pos = np.zeros((n, 3))

    # Copy to avoid mutating input arrays
    x = position.astype(float).copy()
    v = velocity.astype(float).copy()

    for i in range(n):
        pos[i] = x
        vel[i] = v

        # Acceleration = F/m
        a = gravity_force(mass_planet, mass_lander, x) / mass_lander

        # Forward Euler update
        x = x + dt * v
        v = v + dt * a

    return t_array, pos, vel


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


