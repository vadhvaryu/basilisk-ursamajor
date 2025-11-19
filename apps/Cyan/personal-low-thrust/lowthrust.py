"""
Low-thrust continuous transfer from LEO to GEO, using:
1 thruster, opposite direction of velocity, stop thrust when orbit reaches GEO, then coast for 2 weeks, check stability of orbit.

+ Plot trajectory and evaluate circularity after coast.
"""

import numpy as np
import matplotlib.pyplot as plt
from Basilisk.utilities import SimulationBaseClass, macros from Basilisk.simulation import spacecraft, simIncludeGravBody

def run_lowthrust_basilisk(): 
    # --- Constants --- 
    mu = 3.986e14 # m^3/s^2 
    R_E = 6.378e6 # m, Earth radius

    # --- Initial / Target Orbit --- 
    a_init = 500e3 # m, initial altitude above Earth 
    r_init = R_E + a_init v_init = np.sqrt(mu / r_init)
    
    a_final = 35_000e3 # m, final altitude (~GEO) 
    r_target = R_E + a_final

    # --- Spacecraft Mass --- 
    m0 = 1000.0 # kg

    # --- Engine Parameters --- 
    T = 1.0 # N, max thrust 
    I_sp = 10_000 # s 
    g0 = 9.807 # m/s^2 
    
    # --- Spacecraft State --- 
    r_vec = np.array([r_init, 0, 0]) 
    v_vec = np.array([0, v_init, 0]) 
    m = m0

    # === TIME PARAMETERS ===
    dt = 10.0                                   # 10 second time step
    t = 0.0
    t_max = 1e8                              # hard stop (safety)

    # === LOGGING ===
    r_hist = []
    v_hist = []
    m_hist = []
    t_hist = []

    # --- Thrust Effector --- 
    class LowThrustEffector(): 
        def __init__(self, r_vec, v_vec, m, T, I_sp, g0, r_target): 
            self.r_vec = r_vec 
            self.v_vec = v_vec 
            self.m = m 
            self.T = T 
            self.I_sp = I_sp 
            self.g0 = g0 
            self.r_target = r_target 
            self.active = True

        def update(self, dt): 
            r_mag = np.linalg.norm(self.r_vec) 
            v_mag = np.linalg.norm(self.v_vec) 
            if self.active and r_mag < self.r_target: 
                # Acceleration opposite velocity 
                a_thrust = (self.T / self.m) * (-self.v_vec / v_mag) 
                self.v_vec += a_thrust * dt 
                self.m -= (self.T / (self.I_sp * self.g0)) * dt 
            else: 
                self.active = False
    lowthrust = LowThrustEffector(r_vec, v_vec, m, T, I_sp, g0, r_target)

    # --- Thrust Phase Simulation --- 
    while t < t_max and lowthrust.active: 
        # Update thrust 
        lowthrust.update(dt) 
        # Gravity update 
        r_mag = np.linalg.norm(lowthrust.r_vec) 
        acc_grav = -mu * lowthrust.r_vec / r_mag**3 
        lowthrust.v_vec += acc_grav * dt lowthrust.r_vec += lowthrust.v_vec * dt 
        # Logging 
        r_hist.append(lowthrust.r_vec.copy()) 
        v_hist.append(lowthrust.v_vec.copy()) 
        m_hist.append(lowthrust.m) 
        t_hist.append(t) 
        t += dt
    # Save thrust phase end states 
    r_vec_thrust_end = lowthrust.r_vec.copy() 
    v_vec_thrust_end = lowthrust.v_vec.copy() 
    m_thrust_end = lowthrust.m 
    t_thrust_end = t

    print("Thrust phase complete at radius (m):", r_vec_thrust_end) 
    print("Velocity (m/s):", v_vec_thrust_end) 
    print("Remaining mass (kg):", m_thrust_end)

    # --- Coast Phase --- 
    coast_duration = 14*24*3600 # 2 weeks 
    coast_steps = int(coast_duration / dt) 
    for _ in range(coast_steps): 
        r_mag = np.linalg.norm(lowthrust.r_vec) 
        acc_grav = -mu * lowthrust.r_vec / r_mag**3 
        lowthrust.v_vec += acc_grav * dt 
        lowthrust.r_vec += lowthrust.v_vec * dt 
        # Logging 
        r_hist.append(lowthrust.r_vec.copy()) 
        v_hist.append(lowthrust.v_vec.copy())
        m_hist.append(lowthrust.m) 
        t_hist.append(t) t += dt
    
# --- Convert logs to arrays --- 
    r_hist = np.array(r_hist) 
    v_hist = np.array(v_hist) 
    m_hist = np.array(m_hist) 
    t_hist = np.array(t_hist)

# --- Coast orbit analysis --- 
    r_per = np.min(np.linalg.norm(r_hist[-coast_steps:], axis=1)) 
    r_apo = np.max(np.linalg.norm(r_hist[-coast_steps:], axis=1)) 
    e_coast = (r_apo - r_per)/(r_apo + r_per) 
    r_coast_avg = (r_apo + r_per)/2 
    r_error = (r_coast_avg - r_target)/r_target * 100

# --- Summary Text --- 
summary_text = ( 
    f"Initial Altitude: {a_init/1e3:.1f} km\n" 
    f"Target Altitude: {a_final/1e3:.1f} km\n" 
    f"Max Thrust: {T:.3f} N\n" 
    f"Initial speed: {v_init:.3f} m/s\n" 
    f"Final thrust speed: {np.linalg.norm(v_vec_thrust_end):.3f} m/s\n" 
    f"Propellant used: {m0 - m_thrust_end:.3f} kg\n" 
    f"Thrust phase duration: {t_thrust_end/86400:.2f} days\n" 
    f"Coast avg radius error: {r_error:.3f} %\n" 
    f"Coast eccentricity: {e_coast:.5f}\n" 
    f"Speed after coast: {np.linalg.norm(lowthrust.v_vec):.3f} m/s" 
    ) 
    print(summary_text)

# --- Plotting --- 
    fig, ax = plt.subplots(figsize=(12,12)) 
    ax.set_aspect('equal') 
    ax.axis('off')

# Earth 
    from matplotlib.patches import Circle 
    ax.add_patch(Circle((0,0), R_E, fc='C0', ec='none')) 
    ax.annotate("Earth", xy=(0,0), ha='center', va='center', color='white')
# Target orbit 
    ax.add_patch(Circle((0,0), r_target, fc='none', ec='C1', lw=2, ls='--'))

# Trajectories 
    thrust_idx_end = int(t_thrust_end/dt) 
    r_vec_thrust = r_hist[:thrust_idx_end] 
    r_vec_coast = r_hist[thrust_idx_end:] 
    ax.plot(r_vec_thrust[:,0], r_vec_thrust[:,1], color='C2', lw=1, label='Thrust Phase') 
    ax.plot(r_vec_coast[:,0], r_vec_coast[:,1], color='C3', lw=1, label='Coast Phase')

# Mark thrust end / coast start 
    ax.plot(r_vec_thrust[-1,0], r_vec_thrust[-1,1], 'ko', markersize=8, label='Thrust End / Coast Start') 
    ax.annotate("Thrust End / Coast Start", xy=(r_vec_thrust[-1,0], r_vec_thrust[-1,1]), xytext=(r_vec_thrust[-1,0]*1.05, r_vec_thrust[-1,1]*1.05), arrowprops=dict(arrowstyle="->", color='black'), fontsize=10)

# Legend 
    ax.legend(loc='upper right', fontsize=10)

# Summary text 
    ax.text( 
        0.02, 0.98, 
        summary_text, 
        transform=ax.transAxes, 
        fontsize=10, 
        verticalalignment='top', 
        horizontalalignment='left', 
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8) 
        ) 
    plt.show()
if __name__ == "__main__":
    run_lowthrust_basilisk()
