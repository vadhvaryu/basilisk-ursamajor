import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread(r"apps\veer\molniyaorbits\groundtrack.jpg")




mu_earth = 3.986e14
r_earth = 6371e3
omega_earth = 7.2921159e-5

a = 26560e3
e = 0.7
i = np.deg2rad(63.4)
omega = np.deg2rad(270)
Omega = np.deg2rad(0)

T = 2 * np.pi * np.sqrt(a**3 / mu_earth)
print(f"Orbital period: {T/3600:.2f} hr")

num_points = 2000
t = np.linspace(0, T, num_points)
n = np.sqrt(mu_earth / a**3)

M = n * t
E = np.zeros_like(M)
for k in range(len(M)):
    E_guess = M[k]
    for _ in range(10):
        E_guess = M[k] + e * np.sin(E_guess)
    E[k] = E_guess

nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                    np.sqrt(1 - e) * np.cos(E / 2))
r = a * (1 - e * np.cos(E))

x_orb = r * np.cos(nu)
y_orb = r * np.sin(nu)
z_orb = np.zeros_like(x_orb)

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

theta_E = omega_earth * t
x_ecef = np.cos(theta_E) * x_eci + np.sin(theta_E) * y_eci
y_ecef = -np.sin(theta_E) * x_eci + np.cos(theta_E) * y_eci
z_ecef = z_eci

lat = np.rad2deg(np.arcsin(z_ecef / np.linalg.norm([x_ecef, y_ecef, z_ecef], axis=0)))
lon = np.rad2deg(np.arctan2(y_ecef, x_ecef))
lon = (lon + 180) % 360 - 180

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(img, extent=[-180, 180, -90, 90], aspect='auto', zorder=0)
ax.plot(lon, lat, 'gold', linewidth=1.5, zorder=1)
ax.set_title("Molniya Orbit Ground Track (One Orbit ≈ 12 hr)")
ax.set_xlabel("Longitude [°]")
ax.set_ylabel("Latitude [°]")
ax.grid(True, zorder=2)
ax.axhline(0, color='k', linewidth=0.5, zorder=2)
ax.axvline(0, color='k', linewidth=0.5, zorder=2)
ax.set_xlim([-180, 180])
ax.set_ylim([-90, 90])

plt.tight_layout()
plt.show()
