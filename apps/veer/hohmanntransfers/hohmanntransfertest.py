import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Constants
# -------------------------------------------------------------
mu_earth = 3.986e14             # Earth's gravitational parameter [m^3/s^2]
r_earth = 6371e3                # Earth's mean radius [m]
r_leo = r_earth + 500e3         # LEO altitude [m]
r_geo = r_earth + 35786e3       # GEO altitude [m]

# -------------------------------------------------------------
# Hohmann transfer calculations
# -------------------------------------------------------------
a_transfer = (r_leo + r_geo) / 2                     # Semi-major axis [m]
v_leo = np.sqrt(mu_earth / r_leo)                    # Velocity in LEO [m/s]
v_geo = np.sqrt(mu_earth / r_geo)                    # Velocity in GEO [m/s]
v_perigee = np.sqrt(mu_earth * (2/r_leo - 1/a_transfer))  # Vel at perigee [m/s]
v_apogee = np.sqrt(mu_earth * (2/r_geo - 1/a_transfer))   # Vel at apogee [m/s]

delta_v1 = v_perigee - v_leo
delta_v2 = v_geo - v_apogee
time_of_flight = np.pi * np.sqrt(a_transfer**3 / mu_earth)  # seconds

print(f"Δv₁ = {delta_v1:.2f} m/s, Δv₂ = {delta_v2:.2f} m/s, ToF = {time_of_flight/3600:.2f} hr")

# -------------------------------------------------------------
# Orbit geometry for plotting
# -------------------------------------------------------------
theta = np.linspace(0, 2*np.pi, 500)

# LEO and GEO circular orbits
x_leo = r_leo * np.cos(theta)
y_leo = r_leo * np.sin(theta)
x_geo = r_geo * np.cos(theta)
y_geo = r_geo * np.sin(theta)

# Transfer ellipse
e_transfer = (r_geo - r_leo) / (r_geo + r_leo)        # Eccentricity
r_transfer = a_transfer * (1 - e_transfer**2) / (1 + e_transfer * np.cos(theta))
x_transfer = r_transfer * np.cos(theta)
y_transfer = r_transfer * np.sin(theta)

# -------------------------------------------------------------
# Plot
# -------------------------------------------------------------
plt.figure(figsize=(7, 7))
plt.plot(x_transfer/1000, y_transfer/1000, 'b-', label='Transfer Orbit')
plt.plot(x_leo/1000, y_leo/1000, 'g--', label='LEO (500 km)')
plt.plot(x_geo/1000, y_geo/1000, 'r--', label='GEO (35,786 km)')
plt.plot(0, 0, 'yo', label='Earth')

plt.title("Hohmann Transfer: 500 km LEO → GEO")
plt.xlabel("X [km]")
plt.ylabel("Y [km]")
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
