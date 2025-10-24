import numpy as np
import matplotlib.pyplot as plt

mu_earth = 3.986e14
r_earth = 6371e3

a = 26560e3
e = 0.7
i = np.deg2rad(63.4)
omega = np.deg2rad(270)
Omega = np.deg2rad(0)

T = 2 * np.pi * np.sqrt(a**3 / mu_earth)
print(f"Orbital Period = {T/3600:.2f} hr")

theta = np.linspace(0, 2 * np.pi, 800)
r = a * (1 - e**2) / (1 + e * np.cos(theta))

x_orb = r * np.cos(theta)
y_orb = r * np.sin(theta)
z_orb = np.zeros_like(theta)

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

pos_orb = np.vstack((x_orb, y_orb, z_orb))
pos_eci = R @ pos_orb
x_eci, y_eci, z_eci = pos_eci

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(x_eci / 1000, y_eci / 1000, z_eci / 1000, 'b', label='Molniya Orbit')
ax.scatter(0, 0, 0, color='y', label='Earth')

ax.set_xlabel('X [km]')
ax.set_ylabel('Y [km]')
ax.set_zlabel('Z [km]')
ax.set_title('Molniya Orbit (a=26,560 km, e=0.7, i=63.4°)')
ax.legend()
ax.set_box_aspect([1, 1, 1])
ax.grid(True)

plt.tight_layout()
plt.show()
