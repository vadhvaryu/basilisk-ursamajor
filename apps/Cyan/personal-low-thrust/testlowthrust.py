"""
fixed_lowthrust.py

Low-thrust LEO -> GEO transfer (Basilisk v2.8.19 friendly)
- Single, continuous thrust aligned with instantaneous velocity vector
- Stop thrust when radius >= GEO radius
- Simulate a 2-week coast and analyze the final orbit
- Records trajectory and makes plots

Usage:
  1) Make sure Basilisk is built/installed into your active venv (see printed error if not)
  2) From PowerShell with venv active:
       cd C:\Users\Coral14\Documents\basilisk-ursamajor\apps\Cyan\personal-low-thrust
       python fixed_lowthrust.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from math import sqrt

# -------------------------
# Try to find Basilisk automatically (dist3 path used in this repo)
# -------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))  # goes up to Documents/basilisk-ursamajor
auto_bsk_path = os.path.join(repo_root, "basilisk", "dist3", "Basilisk")

if os.path.isdir(auto_bsk_path) and auto_bsk_path not in sys.path:
    sys.path.insert(0, auto_bsk_path)

# If import fails, print helpful message and exit
try:
    from Basilisk.utilities import SimulationBaseClass, macros, unitTestSupport
    from Basilisk.simulation import spacecraft, extForceTorque
    from Basilisk.utilities import simIncludeGravBody
    from Basilisk.utilities import simIncludeGravBody as grav_helper  # alias
    from Basilisk.architecture import messaging
except Exception as e:
    print("ERROR importing Basilisk:", e)
    print("\nThe script expects a built/installed Basilisk Python package (dist3).")
    print("If you haven't built Basilisk, do from the basilisk folder:")
    print("  cd <repo>/basilisk")
    print("  python conanfile.py")
    print("Then activate your venv and re-run this script.")
    sys.exit(1)

# -------------------------
# Physical & simulation parameters (SI)
# -------------------------
MU = 3.986004418e14       # Earth mu (m^3/s^2)
R_E = 6.378e6             # Earth radius (m)

ALT_LEO = 500e3           # 500 km altitude
R_LEO = R_E + ALT_LEO
V_LEO = sqrt(MU / R_LEO)

ALT_GEO = 35_786e3        # GEO altitude
R_GEO = R_E + ALT_GEO

M0 = 1000.0               # initial mass (kg)
T_MAX = 200.0             # N (testing at 200 N)
Isp = 1e4                 # s (large Isp as placeholder)
g0 = 9.80665              # m/s^2

CONTROL_DT = 10.0         # control/update step (s)
SIM_DT = 1.0              # physics timestep (s) used to create the Basilisk task
COAST_DURATION = 14 * 24 * 3600.0  # 2 weeks in seconds
SAFETY_MAX = 3.0 * 365.0 * 24.0 * 3600.0  # 3 years safety (won't be hit)

# -------------------------
# Helper
# -------------------------
def unit(vec):
    n = np.linalg.norm(vec)
    return vec / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])

# -------------------------
# Main scenario
# -------------------------
def run(show_plots=True):
    print("Starting Basilisk low-thrust scenario (fixed_lowthrust.py)")
    print(f"Initial orbit: {ALT_LEO/1e3:.1f} km  -> Target GEO altitude: {ALT_GEO/1e3:.1f} km")
    print(f"Tmax = {T_MAX} N, Isp = {Isp} s, m0 = {M0} kg\n")

    # Create simulation container
    scSim = SimulationBaseClass.SimBaseClass()

    # Create process + task using a consistent physics timestep
    dynProcess = scSim.CreateNewProcess("dynProcess")
    taskRate = macros.sec2nano(SIM_DT)
    dynProcess.addTask(scSim.CreateNewTask("dynTask", taskRate))

    # Create spacecraft and set mass + initial states (inertial N-frame)
    sc = spacecraft.Spacecraft()
    sc.ModelTag = "spacecraft"
    sc.hub.mHub = M0
    sc.hub.r_CN_NInit = [R_LEO, 0.0, 0.0]
    sc.hub.v_CN_NInit = [0.0, V_LEO, 0.0]

    scSim.AddModelToTask("dynTask", sc)

    # Gravity body (simple point mass Earth via factory)
    gravFactory = simIncludeGravBody.gravBodyFactory()
    earth = gravFactory.createEarth()
    earth.isCentralBody = True
    # attach to spacecraft (safe multi-version approach)
    try:
        gravFactory.addBodiesTo(sc)
    except Exception:
        try:
            sc.gravField.gravBodies = [earth]
        except Exception:
            pass

    # External force effector: we use ExtForceTorque to apply arbitrary force in N-frame
    extForce = extForceTorque.ExtForceTorque()
    extForce.ModelTag = "thrustEffector"
    # Add the dynamic effector to the spacecraft and task
    sc.addDynamicEffector(extForce)
    scSim.AddModelToTask("dynTask", extForce)

    # Setup logger for spacecraft state (scStateOutMsg)
    state_log = sc.scStateOutMsg.recorder(taskRate)
    scSim.AddModelToTask("dynTask", state_log)

    # Initialize simulation
    scSim.InitializeSimulation()

    # MAIN LOOP: throttle until radius >= GEO, then coast for COAST_DURATION
    t_sim = 0.0
    thrusting = True
    thrust_end_time = None
    step = 0
    max_steps = int(min(SAFETY_MAX / CONTROL_DT, 50_000_000))

    r_hist = []
    v_hist = []
    m_hist = []
    t_hist = []

    print("Running thrust phase...")
    while step < max_steps:
        # read current state from sc (use scStateOutMsg)
        try:
            scState = sc.scStateOutMsg.read()
            r_N = np.array(scState.r_BN_N)
            v_N = np.array(scState.v_BN_N)
            mass = sc.hub.mHub
        except Exception:
            # fallback to hub initial fields (shouldn't happen after Initialize)
            r_N = np.array(sc.hub.r_CN_NInit)
            v_N = np.array(sc.hub.v_CN_NInit)
            mass = sc.hub.mHub

        r_mag = np.linalg.norm(r_N)

        # Stop thrust when we reach GEO radius (>=)
        if thrusting and r_mag >= R_GEO:
            thrusting = False
            thrust_end_time = t_sim
            # zero force
            extForce.extForce_N = [[0.0], [0.0], [0.0]]
            print(f"Thrust cutoff: reached GEO radius at t = {t_sim/3600.0:.4f} hr ({t_sim/86400.0:.6f} days).")
            # record current state, then break to coast loop
            r_hist.append(r_N.copy())
            v_hist.append(v_N.copy())
            m_hist.append(mass)
            t_hist.append(t_sim)
            break

        # If still thrusting -> compute thrust in direction of velocity vector (tangential)
        if thrusting:
            v_hat = unit(v_N)
            F_N = T_MAX * v_hat  # Newtons (in inertial frame)
            extForce.extForce_N = [[float(F_N[0])], [float(F_N[1])], [float(F_N[2])]]

            # mass flow for this CONTROL_DT: dm = -T / (Isp * g0) * dt
            dm = - (T_MAX / (Isp * g0)) * CONTROL_DT
            sc.hub.mHub = max(0.1, sc.hub.mHub + dm)
        else:
            # ensure no thrust
            extForce.extForce_N = [[0.0], [0.0], [0.0]]

        # Advance simulation by CONTROL_DT seconds using ConfigureStopTime + ExecuteSimulation
        scSim.ConfigureStopTime(macros.sec2nano(t_sim + CONTROL_DT))
        scSim.ExecuteSimulation()

        # After stepping, read and log the state
        scState = sc.scStateOutMsg.read()
        r_N = np.array(scState.r_BN_N)
        v_N = np.array(scState.v_BN_N)
        mass = sc.hub.mHub

        r_hist.append(r_N.copy())
        v_hist.append(v_N.copy())
        m_hist.append(mass)
        t_hist.append(t_sim)

        t_sim += CONTROL_DT
        step += 1

        # quick progress print
        if step % 1000 == 0:
            print(f"  step {step}, t = {t_sim/86400.0:.4f} days, r = {np.linalg.norm(r_N)/1e3:.1f} km, m = {mass:.2f} kg")

        # safety break if time is crazy
        if t_sim >= SAFETY_MAX:
            print("Safety time reached, stopping.")
            break

    # Now run coast for 2 weeks (if thrust_end_time is set)
    if thrust_end_time is None:
        thrust_end_time = t_sim

    coast_stop_time = thrust_end_time + COAST_DURATION
    coast_sample = 600.0  # sample every 10 minutes
    print("Running coast phase for 2 weeks...")

    while t_sim < coast_stop_time:
        next_t = min(t_sim + coast_sample, coast_stop_time)
        scSim.ConfigureStopTime(macros.sec2nano(next_t))
        scSim.ExecuteSimulation()

        scState = sc.scStateOutMsg.read()
        r_N = np.array(scState.r_BN_N)
        v_N = np.array(scState.v_BN_N)
        mass = sc.hub.mHub

        r_hist.append(r_N.copy())
        v_hist.append(v_N.copy())
        m_hist.append(mass)
        t_hist.append(t_sim)

        t_sim = next_t

    # Convert logs to numpy arrays
    r_hist = np.array(r_hist)
    v_hist = np.array(v_hist)
    m_hist = np.array(m_hist)
    t_hist = np.array(t_hist)

    # Basic orbit analysis on the coast phase
    # select coast samples (t >= thrust_end_time)
    coast_mask = t_hist >= thrust_end_time
    if coast_mask.any():
        coast_r = np.linalg.norm(r_hist[coast_mask], axis=1)
        r_apo = coast_r.max()
        r_per = coast_r.min()
        ecc = (r_apo - r_per) / (r_apo + r_per) if (r_apo + r_per) != 0 else 0.0
        r_avg = 0.5 * (r_apo + r_per)
        rel_err_pct = (r_avg - R_GEO) / R_GEO * 100.0
    else:
        r_apo = r_per = r_avg = ecc = rel_err_pct = np.nan

    print("\n=== Summary ===")
    print(f"Initial altitude (km): {(R_LEO - R_E)/1e3:.3f}")
    print(f"Target GEO altitude (km): {ALT_GEO/1e3:.3f}")
    print(f"Final mass (kg): {sc.hub.mHub:.3f}")
    print(f"Perigee after coast (km): {(r_per - R_E)/1e3:.3f}")
    print(f"Apogee after coast  (km): {(r_apo - R_E)/1e3:.3f}")
    print(f"Eccentricity after coast: {ecc:.6f}")
    print(f"Avg radius error vs GEO: {rel_err_pct:.6f} %")

    # Plot trajectory (XY)
    if show_plots and len(r_hist) > 0:
        fig, ax = plt.subplots(figsize=(9,9))
        ax.set_aspect('equal')
        # Earth
        earth_patch = Circle((0,0), R_E/1e3, color='C0', zorder=0)
        ax.add_patch(earth_patch)
        # GEO circle
        theta = np.linspace(0, 2*np.pi, 400)
        ax.plot((R_GEO/1e3)*np.cos(theta), (R_GEO/1e3)*np.sin(theta),
                '--', lw=1.5, label='GEO radius')
        # trajectory
        ax.plot(r_hist[:,0]/1e3, r_hist[:,1]/1e3, lw=1, label='Trajectory')
        ax.scatter(r_hist[0,0]/1e3, r_hist[0,1]/1e3, color='g', label='Start')
        if thrust_end_time is not None:
            # find index near thrust_end_time
            idx = np.searchsorted(t_hist, thrust_end_time)
            if idx < len(r_hist):
                ax.scatter(r_hist[idx,0]/1e3, r_hist[idx,1]/1e3, color='k', label='Thrust end')
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.legend()
        ax.set_title('LEO->GEO Low-Thrust Trajectory (XY)')
        plt.show()

    return {
        "r_hist": r_hist,
        "v_hist": v_hist,
        "t_hist": t_hist,
        "m_hist": m_hist,
        "thrust_end_time": thrust_end_time,
        "summary": {
            "r_apo": r_apo,
            "r_per": r_per,
            "ecc": ecc,
            "r_avg": r_avg,
            "rel_err_pct": rel_err_pct
        }
    }

if __name__ == "__main__":
    run(show_plots=True)
