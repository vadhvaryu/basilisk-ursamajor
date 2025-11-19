"""
Low-thrust continuous transfer from LEO to GEO, using:
1 thruster, opposite direction of velocity, stop thrust when orbit reaches GEO, then coast for 2 weeks, check stability of orbit.

+ Plot trajectory and evaluate circularity after coast.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def run_lowthrust_basilisk():

    # === CONSTANTS ===
    mu = 3.986004418e14         # Earth gravitational parameter (m^3/s^2)
    R_E = 6.378e6               # Earth radius (m)
    g0 = 9.80665                # m/s^2

    # === INITIAL CONDITIONS ===
    alt_init = 500e3                            # 500 km LEO
    r0 = R_E + alt_init
    v0 = np.sqrt(mu / r0)                       # Circular orbit speed

    r_vec = np.array([r0, 0.0, 0.0])
    v_vec = np.array([0.0, v0, 0.0])
    m0 = 1000.0                                 # kg

    # === TARGET ORBIT ===
    alt_final = 35_786e3                        # GEO altitude
    r_target = R_E + alt_final                  # GEO radius (~42164 km)

    # === THRUSTER PARAMETERS ===
    T = 1.0                                     # N
    Isp = 10_000                                # s

    # === TIME PARAMETERS ===
    dt = 10.0                                   # 10 second time step
    t = 0.0
    t_max = 2e7                                 # hard stop (safety)

    # === LOGGING ===
    r_hist = []
    v_hist = []
    m_hist = []
    t_hist = []

    # === Helper: compute semi-major axis from r, v ===
    def semi_major_axis(r, v):
        rmag = np.linalg.norm(r)
        vmag = np.linalg.norm(v)
        return 1.0 / (2.0 / rmag - vmag**2 / mu)

    # === Helper: thrust update ===
    def apply_thrust(r, v, m, dt):
        vmag = np.linalg.norm(v)
        if vmag < 1e-8:
            return r, v, m    # no thrust if velocity is zero

        thrust_dir = v / vmag                # spacecraft accelerates forward
        a_thrust = (T / m) * thrust_dir      # acceleration

        # update velocity
        v_new = v + a_thrust * dt

        # update mass
        m_new = m - (T / (Isp * g0)) * dt
        return r, v_new, m_new

    # === THRUST PHASE LOOP ===
    thrust_active = True

    while t < t_max and thrust_active:

        # 1) apply thrust
        r_vec, v_vec, m0 = apply_thrust(r_vec, v_vec, m0, dt)

        # 2) gravity update
        rmag = np.linalg.norm(r_vec)
        a_grav = -mu * r_vec / rmag**3
        v_vec += a_grav * dt
        r_vec += v_vec * dt

        # 3) compute orbital energy
        a_current = semi_major_axis(r_vec, v_vec)

        # === STOP THRUST WHEN TARGET a REACHED ===
        if a_current >= r_target:
            thrust_active = False
            print("\n=== THRUST PHASE COMPLETE ===")
            print("Semi-major axis reached GEO.")
            print(f"Time elapsed: {t/86400:.2f} days")
            print(f"Remaining mass: {m0:.2f} kg")

        # Logging
        r_hist.append(r_vec.copy())
        v_hist.append(v_vec.copy())
        m_hist.append(m0)
        t_hist.append(t)

        t += dt


    # === BEGIN COAST PHASE (2 weeks) ===
    coast_duration = 14 * 24 * 3600       # 2 weeks in seconds
    coast_steps = int(coast_duration / dt)

    for _ in range(coast_steps):

        rmag = np.linalg.norm(r_vec)
        a_grav = -mu * r_vec / rmag**3
        v_vec += a_grav * dt
        r_vec += v_vec * dt

        r_hist.append(r_vec.copy())
        v_hist.append(v_vec.copy())
        m_hist.append(m0)
        t_hist.append(t)

        t += dt

    r_hist = np.array(r_hist)
    v_hist = np.array(v_hist)


    # === COAST ORBIT ANALYSIS ===
    r_last = np.linalg.norm(r_hist[-coast_steps:], axis=1)
    r_per = np.min(r_last)
    r_apo = np.max(r_last)
    e = (r_apo - r_per) / (r_apo + r_per)
    r_avg = 0.5 * (r_apo + r_per)
    rel_error = (r_avg - r_target) / r_target * 100

    print("\n=== COAST PERIOD ANALYSIS ===")
    print(f"Perigee:  {r_per/1e3:.2f} km")
    print(f"Apogee:   {r_apo/1e3:.2f} km")
    print(f"Eccentricity: {e:.6f}")
    print(f"Avg. radius error vs GEO: {rel_error:.4f}%")

    # === GRAPH TRAJECTORY ===
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal')
    ax.axis('off')

    # Earth
    ax.add_patch(Circle((0, 0), R_E, fc='C0'))
    ax.annotate("Earth", xy=(0, 0), ha='center', va='center', color='white')

    # Target orbit
    ax.add_patch(Circle((0, 0), r_target, fc='none', ec='C1', lw=2, ls='--'))

    # Trajectory
    thrust_end_idx = len(t_hist) - coast_steps
    ax.plot(r_hist[:thrust_end_idx, 0], r_hist[:thrust_end_idx, 1],
            color='C2', lw=1, label='Thrust Phase')
    ax.plot(r_hist[thrust_end_idx:, 0], r_hist[thrust_end_idx:, 1],
            color='C3', lw=1, label='Coast Phase')

    # Mark thrust end
    ax.plot(r_hist[thrust_end_idx, 0], r_hist[thrust_end_idx, 1],
            'ko', markersize=8)
    ax.annotate("Thrust End",
                xy=(r_hist[thrust_end_idx, 0], r_hist[thrust_end_idx, 1]),
                xytext=(1.05 * r_hist[thrust_end_idx, 0],
                        1.05 * r_hist[thrust_end_idx, 1]),
                arrowprops=dict(arrowstyle="->"))

    ax.legend()
    plt.savefig("lowthrust_graph_fixed.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    run_lowthrust_basilisk()
