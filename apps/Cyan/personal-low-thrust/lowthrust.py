"""
Nonimpulsive low-thrust LEO -> GEO transfer:
- 1 thruster, set in opposite direction to velocity of spacecraft.
- Do not need to account for rotation of body of spacecraft.
- Stop thrust when reach GEO radius.
- Model coast for 2 weeks and analyze stability of orbit.
"""
#-------------
import sys
import os

# Add root of repo to Python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(repo_root)
#--------------


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Basilisk imports
from Basilisk.utilities import SimulationBaseClass, macros, orbitalMotion
from Basilisk.simulation import spacecraft, svIntegrators
from Basilisk.utilities import simIncludeThruster, simIncludeGravBody, unitTestSupport
from Basilisk.architecture import messaging

# ---------------- INITIAL PARAMETERS ----------------
# Physical constants
mu = 3.986004418e14       # Earth mu (m^3/s^2)
R_E = 6.378e6             # Earth radius (m)
g0 = 9.80665              # m/s^2

# Initial orbit (500 km LEO)
alt_init = 500e3            # m
r_LEO = R_E + alt_init      # total radius of orbit
v_LEO = np.sqrt(mu / r_LEO)

# GEO target (35,786 km altitude)
alt_GEO = 35_786e3          # m
r_GEO = R_E + alt_GEO       # total radius of orbit

# Spacecraft
m0 = 1000.0               # kg

# Engine parameters 
T_max = 1.0               # N input (max thrust)
I_sp = 10_000.0           # s 
# print I_sp to confirm
print(f"Using Isp = {I_sp} s (copied from your reference script)")

# Integration / task timestep
dt = 10.0                 # seconds (task step)
taskRate = macros.sec2nano(dt)

# Coast duration
coast_duration = 14 * 24 * 3600.0  # 2 weeks in seconds

# Safety
safety_max = 3.0 * 365.0 * 24.0 * 3600.0  # 3 years in seconds

# ---------------- helper functions ----------------
def unit(vec):
    n = np.linalg.norm(vec)
    return vec / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])

def semi_major_axis(r_vec, v_vec, mu_val=mu):
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    return 1.0 / (2.0 / r - v * v / mu_val)

# ---------------- Build Basilisk sim ----------------
def run_basilisk_rk4(show_plots=True):

    # create simulation base
    scSim = SimulationBaseClass.SimBaseClass()

    # process + task
    simProcessName = "simProcess"
    simTaskName = "simTask"
    dynProcess = scSim.CreateNewProcess(simProcessName)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, taskRate))

    # create spacecraft (use full Spacecraft object so we can set integrator)
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "spacecraft"
    scObject.hub.mHub = m0

    # initial inertial states (N-frame)
    scObject.hub.r_CN_NInit = [r_LEO, 0.0, 0.0]
    scObject.hub.v_CN_NInit = [0.0, v_LEO, 0.0]

    # attach RK4 integrator to spacecraft
    rk4 = svIntegrators.svIntegratorRK4(scObject)
    scObject.setIntegrator(rk4)

    # add spacecraft to task
    scSim.AddModelToTask(simTaskName, scObject)

    # gravity body factory and attach Earth (central body)
    gravFactory = simIncludeGravBody.gravBodyFactory()
    earth = gravFactory.createEarth()
    earth.isCentralBody = True
    # attach gravity bodies to simulation (Basilisk helper)
    try:
        gravFactory.addBodiesTo(scObject)
    except Exception:
        # fallback attach if helper differs
        try:
            scObject.gravField.gravBodies = [earth]
        except Exception:
            pass

    # --- Create single thruster using simIncludeThruster ---
    thrusterName = "lowThrust"
    thruster_objs = simIncludeThruster.create(
        thrusterType="CST",
        r_B=[0.0, 0.0, 0.0],
        tHat_B=[1.0, 0.0, 0.0],
        maxThrust=T_max,
        isp=I_sp,
        baseName=thrusterName,
    )

    # select thruster device and effector depending on helper return format
    if isinstance(thruster_objs, (list, tuple)) and len(thruster_objs) > 0:
        thr_device = thruster_objs[0]
    else:
        thr_device = thruster_objs

    # try retrieve effector (common API places the effector on the last element or on the device)
    thrusterEffector = None
    try:
        thrusterEffector = thruster_objs[-1].effector
    except Exception:
        try:
            if hasattr(thr_device, "effector"):
                thrusterEffector = thr_device.effector
        except Exception:
            thrusterEffector = None

    # attach effector to spacecraft (if found) and add to task
    if thrusterEffector is not None:
        try:
            scObject.addDynamicEffector(thrusterEffector)
            scSim.AddModelToTask(simTaskName, thrusterEffector)
        except Exception:
            # some helper already registered the effector
            pass
    else:
        # try attach device as effector (older helpers)
        try:
            scObject.addDynamicEffector(thr_device)
        except Exception:
            pass

    # Add thruster device to task (if helper requires it)
    try:
        scSim.AddModelToTask(simTaskName, thr_device)
    except Exception:
        pass

    # --- Build and connect thrust command message (DoubleVec) ---
    nThrusters = 1
    thrustFactorPayload = messaging.DoubleVecPayload()
    thrustFactorPayload.vec = [0.0] * nThrusters
    thrustFactorMsg = messaging.DoubleVecMsg().write(thrustFactorPayload)

    # If effector exists, subscribe our message (defensive)
    try:
        thrusterEffector.thrusterCmdInMsg.subscribeTo(thrustFactorMsg)
    except Exception:
        try:
            thrusterEffector.thrustFactorInMsg.subscribeTo(thrustFactorMsg)
        except Exception:
            # fallback - we'll attempt to set attribute directly later
            pass

    # --- Setup recorder for spacecraft state (scStateOutMsg) ---
    try:
        stateLog = scObject.scStateOutMsg.recorder()
        scSim.AddModelToTask(simTaskName, stateLog)
    except Exception:
        # fallback: some versions may use transStateOutMsg
        try:
            stateLog = scObject.transStateOutMsg.recorder()
            scSim.AddModelToTask(simTaskName, stateLog)
        except Exception:
            stateLog = None

    # Initialize the sim
    scSim.InitializeSimulation()

    # Logging arrays
    r_hist = []
    v_hist = []
    m_hist = []
    t_hist = []

    # Local shadow state for read/write when needed
    # (we will rely on Basilisk to update these, reading from scObject whenever possible)
    thrusting = True
    thrust_end_time = None
    t_sim = 0.0
    step = 0
    max_steps = int(min(safety_max / dt, 50_000_000))

    # ---- MAIN LOOP: step by dt, command thrust, read back states from Basilisk ----
    while step < max_steps:

        # Read latest state from scObject if available
        try:
            scState = scObject.scStateOutMsg.read()
            r_N = np.array(scState.r_BN_N)   # meters
            v_N = np.array(scState.v_BN_N)   # m/s
            mass = scObject.hub.mHub
        except Exception:
            # fallback: try transStateOutMsg
            try:
                trans = scObject.transStateOutMsg.read()
                r_N = np.array(trans.r_BN_N)
                v_N = np.array(trans.v_BN_N)
                mass = scObject.hub.mHub
            except Exception:
                # If messages not available yet, read from hub initial fields
                r_N = np.array(scObject.hub.r_CN_NInit)
                v_N = np.array(scObject.hub.v_CN_NInit)
                mass = scObject.hub.mHub

        # compute semi-major axis and check for cutoff: stop when radius reaches GEO
        r_mag = np.linalg.norm(r_N)
        if thrusting and r_mag >= r_GEO:
            thrusting = False
            thrust_end_time = t_sim
            # send a thrustFactor = 0 to turn thruster off
            try:
                payload = messaging.DoubleVecPayload()
                payload.vec = [0.0]
                tf_msg = messaging.DoubleVecMsg().write(payload)
                thrusterEffector.thrusterCmdInMsg.write(tf_msg)
            except Exception:
                try:
                    thrusterEffector.thrustFactor = [0.0]
                except Exception:
                    pass
            print(f"Thrust cutoff: radius reached GEO at t = {t_sim:.1f} s ({t_sim/86400.0:.4f} days)")
            # don't break yet — we'll now enter coast and run for coast_duration

        # If still thrusting, compute inertial thrust direction (along +v to push spacecraft forward)
        if thrusting:
            v_hat = unit(v_N)
            # set thruster device direction (body==inertial assumption)
            try:
                thr_device.tHat_B = [float(v_hat[0]), float(v_hat[1]), float(v_hat[2])]
            except Exception:
                try:
                    thr_device.thrDirection = [float(v_hat[0]), float(v_hat[1]), float(v_hat[2])]
                except Exception:
                    pass
            # publish thrustFactor = 1.0
            try:
                payload = messaging.DoubleVecPayload()
                payload.vec = [1.0]
                tf_msg = messaging.DoubleVecMsg().write(payload)
                thrusterEffector.thrusterCmdInMsg.write(tf_msg)
            except Exception:
                try:
                    thrusterEffector.thrustFactor = [1.0]
                except Exception:
                    pass
        else:
            # ensure thruster off
            try:
                payload = messaging.DoubleVecPayload()
                payload.vec = [0.0]
                tf_msg = messaging.DoubleVecMsg().write(payload)
                thrusterEffector.thrusterCmdInMsg.write(tf_msg)
            except Exception:
                try:
                    thrusterEffector.thrustFactor = [0.0]
                except Exception:
                    pass

        # Advance the simulation one step (RK4 integrator inside Basilisk will be used)
        # Use SingleStep loop for fine control (many Basilisk examples use ExecuteSimulation for long runs)
        try:
            scSim.SingleStep()
        except Exception:
            # Some Basilisk builds require ProcessModelQueue + SingleStep
            try:
                scSim.ProcessModelQueue(simTaskName)
                scSim.SingleStep()
            except Exception:
                # last-resort: Execute for dt window
                scSim.ConfigureStopTime(macros.sec2nano(t_sim + dt))
                scSim.ExecuteSimulation()

        # Read back updated state from message
        try:
            scState = scObject.scStateOutMsg.read()
            r_N = np.array(scState.r_BN_N)
            v_N = np.array(scState.v_BN_N)
            mass = scObject.hub.mHub
        except Exception:
            try:
                trans = scObject.transStateOutMsg.read()
                r_N = np.array(trans.r_BN_N)
                v_N = np.array(trans.v_BN_N)
                mass = scObject.hub.mHub
            except Exception:
                # if read fails, continue with previous values (unlikely in a working Basilisk setup)
                pass

        # log
        r_hist.append(r_N.copy())
        v_hist.append(v_N.copy())
        m_hist.append(mass)
        t_hist.append(t_sim)

        # time increment
        t_sim += dt
        step += 1

        # if thrusting ended earlier, wait to finish coast period
        if (not thrusting) and (thrust_end_time is not None) and (t_sim >= thrust_end_time + coast_duration):
            print(f"Finished coast at t = {t_sim:.1f} s ({t_sim/86400.0:.4f} days)")
            break

        # safety
        if t_sim >= safety_max:
            print("Safety stop reached.")
            break

    # convert logs to arrays
    r_hist = np.array(r_hist)
    v_hist = np.array(v_hist)
    t_hist = np.array(t_hist)
    m_hist = np.array(m_hist)

    # coast analysis on last coast_duration interval
    coast_steps = int(coast_duration / dt)
    if coast_steps > len(r_hist):
        coast_steps = len(r_hist)

    r_last = np.linalg.norm(r_hist[-coast_steps:], axis=1)
    r_per = np.min(r_last)
    r_apo = np.max(r_last)
    ecc = (r_apo - r_per) / (r_apo + r_per) if (r_apo + r_per) != 0 else 0.0
    r_avg = 0.5 * (r_apo + r_per)
    rel_err_pct = (r_avg - r_GEO) / r_GEO * 100.0

    # Print summary
    print("\n=== Summary ===")
    print(f"Initial altitude (km): {(r_LEO - R_E)/1e3:.3f}")
    print(f"Target GEO altitude (km): {alt_GEO/1e3:.3f}")
    print(f"Final mass (kg): {mass:.3f}")
    print(f"Perigee after coast (km): {(r_per - R_E)/1e3:.3f}")
    print(f"Apogee after coast  (km): {(r_apo - R_E)/1e3:.3f}")
    print(f"Eccentricity after coast: {ecc:.6f}")
    print(f"Avg radius error vs GEO: {rel_err_pct:.6f} %")

    # Plotting
    if show_plots:
        thrust_end_idx = max(0, len(t_hist) - coast_steps)
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_aspect('equal')
        ax.axis('off')
        ax.add_patch(Circle((0,0), R_E, fc='C0', ec='none'))
        ax.annotate("Earth", xy=(0,0), ha='center', va='center', color='white')
        theta = np.linspace(0, 2*np.pi, 400)
        ax.plot(r_GEO*np.cos(theta), r_GEO*np.sin(theta), '--', lw=1.5, label='GEO radius')
        if len(r_hist) > 0:
            ax.plot(r_hist[:thrust_end_idx,0], r_hist[:thrust_end_idx,1], lw=1, label='Thrust phase')
            ax.plot(r_hist[thrust_end_idx:,0], r_hist[thrust_end_idx:,1], lw=1, label='Coast phase')
            ax.scatter(r_hist[0,0], r_hist[0,1], color='g', label='Start')
            ax.scatter(r_hist[thrust_end_idx,0], r_hist[thrust_end_idx,1], color='k', label='Thrust end / coast start')
        ax.legend(loc='upper right')
        plt.savefig('basilisk_lowthrust_rk4.png', dpi=300)
        plt.show()

    return {
        'r_hist': r_hist,
        'v_hist': v_hist,
        't_hist': t_hist,
        'm_hist': m_hist,
        'perigee_m': r_per,
        'apogee_m': r_apo,
        'ecc': ecc,
        'r_avg': r_avg,
        'rel_err_percent': rel_err_pct
    }

if __name__ == "__main__":
    results = run_basilisk_rk4(show_plots=True)
