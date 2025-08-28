import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Functions import euler_gravity, verlet_gravity

def escape_velocity(mass_body: float, distance_from_center: float):
    """Return escape velocity at a given radius from the planet."""
    G = 6.67430e-11
    return (2 * G * mass_body / distance_from_center) ** 0.5

def run_simulation(
    mass_planet=6.42e23,    # Mars mass (kg)
    rad_planet=3.39e6,      # Mars radius (m)
    orbital_radius=5e6,     # Initial distance from center (m)
    factor_escape=1.1,      # Velocity factor relative to escape velocity
    dt=1,                  # Timestep (s)
    t_max=20000          # Simulation time (s)
):
    """Run both Verlet and Euler gravity simulations for hyperbolic escape."""

    # Initial conditions
    initial_velocity = escape_velocity(mass_planet, orbital_radius)
    velocity_mag = factor_escape * initial_velocity

    position = np.array([0.0, orbital_radius, 0.0])   # On +Y axis
    velocity = np.array([velocity_mag, 0.0, 0.0])     # Tangential in +X

    # Run simulations
    t_array, x_array, v_array = verlet_gravity(position, velocity, 1.0, mass_planet, dt, t_max)
    t_array_e, x_array_e, v_array_e = euler_gravity(position, velocity, 1.0, mass_planet, dt, t_max)

    return (x_array, x_array_e, rad_planet)

def plot_trajectories(x_array, x_array_e, rad_planet):
    """Plot Mars, Verlet trajectory, and Euler trajectory in 3D."""
    X, Y, Z = x_array[:, 0], x_array[:, 1], x_array[:, 2]
    X_e, Y_e, Z_e = x_array_e[:, 0], x_array_e[:, 1], x_array_e[:, 2]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Plot trajectories
    ax.plot(X, Y, Z, color='royalblue', linewidth=2, label='Verlet (Hyperbolic Escape)')
    ax.plot(X_e, Y_e, Z_e, color='orangered', linestyle='--', linewidth=1.5, label='Euler (Diverges)')

    # Start + End
    ax.scatter(X[0], Y[0], Z[0], color='green', marker='o', s=60, label='Start')
    ax.scatter(X[-1], Y[-1], Z[-1], color='black', marker='x', s=60, label='End')

    # Draw Mars
    u, v = np.linspace(0, 2*np.pi, 40), np.linspace(0, np.pi, 20)
    x_sphere = rad_planet * np.outer(np.cos(u), np.sin(v))
    y_sphere = rad_planet * np.outer(np.sin(u), np.sin(v))
    z_sphere = rad_planet * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='orange', alpha=0.5, zorder=-1)

    # Equal aspect ratio
    max_range = np.max(np.linalg.norm(x_array, axis=1))
    for axis in [ax.set_xlim, ax.set_ylim, ax.set_zlim]:
        axis([-max_range, max_range])

    # Labels
    ax.set_title('Hyperbolic Escape from Mars', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run simulation with > escape velocity (hyperbolic)
    x_array, x_array_e, rad_planet = run_simulation(factor_escape=1.1)
    plot_trajectories(x_array, x_array_e, rad_planet)
