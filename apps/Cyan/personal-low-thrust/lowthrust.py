"""
Nonimpulsive low-thrust LEO -> GEO transfer:
- 1 thruster opposite to velocity.
- Stop thrust at GEO.
- Coast 2 weeks and analyze orbit.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Add repo root and basilisk/src to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(repo_root)
basilisk_path = os.path.join(repo_root, "basilisk", "src")
sys.path.append(basilisk_path)

# Basilisk v2.8.19 imports
from Basilisk import simulation as sim
from Basilisk import utilities as util
from Basilisk.simulation import spacecraft, svIntegrators
from Basilisk.simulation import simIncludeThruster, simIncludeGravBody
from Basilisk.architecture import messaging

# ---------------- INITIAL PARAMETERS ----------------
mu = 3.986004418e14
R_E = 6.378e6
alt_init = 500e3
r_LEO = R_E + alt_init
v_LEO = np.sqrt(mu / r_LEO)

alt_GEO = 35_786e3
r_GEO = R_E + alt_GEO

m0 = 1000.0
T_max = 1.0
I_sp = 10_000.0

dt = 10.0  # seconds
coast_duration = 14 * 24 * 3600.0
safety_max = 3.0 * 365.0 * 24.0 * 3600.0

# ---------------- helper functions ----------------
def unit(vec):
    n = np.linalg.norm(vec)
    return vec / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])

# ---------------- Build Basilisk sim ----------------
def run_basilisk(show_plots=True):
    # Create simulation base
    scSim = util.SimBaseClass()

    # Process and task
    simProcessName = "simProcess"
    simTaskName = "simTask"
    dynProcess = scSim.CreateNewProcess(simProcessName)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, dt))

    # Spacecraft
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "spacecraft"
    scObject.hub.mHub = m0
    scObject.hub.r_CN_NInit = [r_LEO, 0.0, 0.0]
    scObject.hub.v_CN_NInit = [0.0, v_LEO, 0.0]

    rk4 = svIntegrators.svIntegratorRK4(scObject)
    scObject.setIntegrator(rk4)

    scSim.AddModelToTask(simTaskName, scObject)

    # Gravity
    gravFactory = simIncludeGravBody.gravBodyFactory()
    earth = gravFactory.createEarth()
    earth.isCentralBody = True
    gravFactory.addBodiesTo(scObject)

    # Thruster
    thruster_objs = simIncludeThruster.create(
        thrusterType="CST",
        r_B=[0.0, 0.0, 0.0],
        tHat_B=[1.0, 0.0, 0.0],
        maxThrust=T_max,
        isp=I_sp,
        baseName="lowThrust",
    )
    thr_device = thruster_objs[0]
    thrusterEffector = thr_device.effector
    scObject.addDynamicEffector(thrusterEffector)
    scSim.AddModelToTask(simTaskName, thrusterEffector)
    scSim.AddModelToTask(simTaskName, thr_device)

    # Command message
    thrustFactorPayload = messaging.DoubleVecPayload()
    thrustFactorPayload.vec = [0.0]
    thrustFactorMsg = messaging.DoubleVecMsg().write(thrustFactorPayload)
    thrusterEffector.thrusterCmdInMsg.subscribeTo(thrustFactorMsg)

    # State logging
    stateLog = scObject.scStateOutMsg.recorder()
    scSim.AddModelToTask(simTaskName, stateLog)

    # Initialize
    scSim.InitializeSimulation()

    r_hist, v_hist, m_hist, t_hist = [], [], [], []

    thrusting = True
    thrust_end_time = None
    t_sim = 0.0
    step = 0
    max_steps = int(min(safety_max / dt, 50_000_000))

    while step < max_steps:
        scSim.SingleStep()
        scState = scObject.scStateOutMsg.read()
        r_N = np.array(scState.r_BN_N)
        v_N = np.array(scState.v_BN_N)
        mass = scObject.hub.mHub

        r_mag = np.linalg.norm(r_N)
        if thrusting and r_mag >= r_GEO:
            thrusting = False
            thrust_end_time = t_sim
            thrusterEffector.thrusterCmdInMsg.write(messaging.DoubleVecMsg().write(messaging.DoubleVecPayload()))
            print(f"Thrust cutoff at t = {t_sim:.1f} s")

        if thrusting:
            v_hat = unit(v_N)
            thr_device.tHat_B = v_hat.tolist()
            payload = messaging.DoubleVecPayload()
            payload.vec = [1.0]
            thrusterEffector.thrusterCmdInMsg.write(messaging.DoubleVecMsg().write(payload))
        else:
            payload = messaging.DoubleVecPayload()
            payload.vec = [0.0]
            thrusterEffector.thrusterCmdInMsg.write(messaging.DoubleVecMsg().write(payload))

        # log
        r_hist.append(r_N.copy())
        v_hist.append(v_N.copy())
        m_hist.append(mass)
        t_hist.append(t_sim)

        t_sim += dt
        step += 1

        if (not thrusting) and (thrust_end_time is not None) and (t_sim >= thrust_end_time + coast_duration):
            print(f"Finished coast at t = {t_sim:.1f} s")
            break

    r_hist = np.array(r_hist)
    v_hist = np.array(v_hist)
    m_hist = np.array(m_hist)
    t_hist = np.array(t_hist)

    # Simple summary
    r_last = np.linalg.norm(r_hist[-int(coast_duration/dt):], axis=1)
    print(f"Final mass: {m_hist[-1]:.2f} kg")
    print(f"Min/Max radius during coast: {r_last.min()/1e3:.1f}/{r_last.max()/1e3:.1f} km")

    # Plot
    if show_plots:
        fig, ax = plt.subplots(figsize=(8,8))
        ax.set_aspect('equal')
        ax.add_patch(Circle((0,0), R_E, fc='C0'))
        theta = np.linspace(0, 2*np.pi, 400)
        ax.plot(r_GEO*np.cos(theta), r_GEO*np.sin(theta), '--', lw=1.5)
        ax.plot(r_hist[:,0], r_hist[:,1], lw=1)
        plt.show()

    return r_hist, v_hist, t_hist, m_hist

if __name__ == "__main__":
    run_basilisk()
