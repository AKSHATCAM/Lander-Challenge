import numpy as np
import matplotlib.pyplot as plt

# Load data (assuming 4 columns logged)
time, h, v_actual, v_target = np.loadtxt("C:/Users/aksha/source/repos/LanderMain/autopilot_output.txt", unpack=True)

# Plot velocity vs altitude
plt.figure(figsize=(8,6))
plt.plot(h, v_actual, label="Actual descent rate")
plt.plot(h, v_target, label="Target descent rate", linestyle="--")
plt.xlabel("Altitude (m)")
plt.ylabel("Vertical velocity (m/s)")
plt.title("Mars Lander Descent Profile")
plt.legend()
plt.gca().invert_xaxis()  # so descent (h→0) goes left to right
plt.grid(True)
plt.show()
