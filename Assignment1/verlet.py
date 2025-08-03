import numpy as np
import matplotlib.pyplot as plt

def verlet_oscillator(m=1, k=1, x0=0, v0=1, dt=1, t_max=1000):
    """
    Simulates a harmonic oscillator using the Verlet method.

    Parameters:
        m (float): Mass
        k (float): Spring constant
        x0 (float): Initial position
        v0 (float): Initial velocity
        dt (float): Time step
        t_max (float): Total simulation time

    Returns:
        t_array (ndarray): Time values
        x_array (ndarray): Position values
        v_array (ndarray): Velocity values
    """
    # Time array
    t_array = np.arange(0, t_max + dt, dt)
    n = len(t_array)

    # Initialize arrays
    x_array = np.zeros(n)
    v_array = np.zeros(n)

    # Set initial conditions
    x_array[0] = x0
    v_array[0] = v0

    # First step using Taylor expansion
    a0 = -k * x0 / m
    x_array[1] = x0 + v0 * dt + 0.5 * a0 * dt**2

    # Verlet integration
    for i in range(2, n):
        a = -k * x_array[i - 1] / m
        x_array[i] = 2 * x_array[i - 1] - x_array[i - 2] + dt**2 * a
        v_array[i - 1] = (x_array[i] - x_array[i - 2]) / (2 * dt)

    # Estimate last velocity
    v_array[-1] = (x_array[-1] - x_array[-3]) / (2 * dt)

    return t_array, x_array, v_array

# Run simulation
t, x, v = verlet_oscillator()

# Plotting
plt.figure(figsize=(10, 5))
plt.title("Verlet Integration of Harmonic Oscillator")
plt.plot(t, x, label='Position x(t)', linewidth=1.5)
plt.plot(t, v, label='Velocity v(t)', linewidth=1.2)
plt.xlabel("Time (s)")
plt.ylabel("x and v")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


