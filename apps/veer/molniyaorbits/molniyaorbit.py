import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Constants
# -------------------------------------------------------------
mu_earth = 3.986e14     # [m^3/s^2]
r_earth = 6371e3        # [m]

# -------------------------------------------------------------
# Molniya Orbit Parameters
# -------------------------------------------------------------
a = 26560e3             # semi-major axis [m] (~12 hr period)
e = 0.7                 # eccentricity
i = np.deg2rad(63.4)    # inclination [rad]
omega = np.deg2rad(270) # argument of perigee [rad]
Omega = np.deg2rad(0)   # RAAN [rad]

# Orbital period
T = 2 * np.pi * np.sqrt(a**3 / mu_earth)
print(f"Orbital Period = {T/3600:.2f} hr")

# -------------------------------------------------------------
# Parametric Orbit in Orbital Plane
# -------------------------------------------------------------
theta = np.linspace(0, 2*np.pi, 800)
r = a * (1 - e**2) / (1 + e * np.cos(theta))

# Position in orbital plane
x_orb = r * np.cos(theta)
y_orb = r * np.sin(theta)
z_orb = np.zeros_like(theta)

# -------------------------------------------------------------
# Rotate to Earth-centered inertial (ECI) frame
# -------------------------------------------------------------
# Rotation matrices
R1_Omega = np.array([
    [np.cos(Omega), -np.sin(Omega), 0],
    [np.sin(Omega),  np.cos(Omega), 0],
    [0, 0, 1]
])

R2_i = np.array([
    [1, 0, 0],
    [0, np.cos(i), -np.sin(i)],
    [0, np.sin(i),  np.cos(i)]
])

R3_omega = np.array([
    [np.cos(omega), -np.sin(omega), 0],
    [np.sin(omega),  np.cos(omega), 0],
    [0, 0, 1]
])

R = R1_Omega @ R2_i @ R3_omega

# Apply transformation
pos_orb = np.vstack((x_orb, y_orb, z_orb))
pos_eci = R @ pos_orb
x_eci, y_eci, z_eci = pos_eci

# -------------------------------------------------------------
# Plot
# -------------------------------------------------------------
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot orbit and Earth
ax.plot(x_eci/1000, y_eci/1000, z_eci/1000, 'b', label='Molniya Orbit')
ax.scatter(0, 0, 0, color='y', label='Earth')

# Visual markers
ax.set_xlabel('X [km]')
ax.set_ylabel('Y [km]')
ax.set_zlabel('Z [km]')
ax.set_title('Molniya Orbit (a=26,560 km, e=0.7, i=63.4°)')
ax.legend()
ax.set_box_aspect([1, 1, 1])
ax.grid(True)

plt.tight_layout()
plt.show()
