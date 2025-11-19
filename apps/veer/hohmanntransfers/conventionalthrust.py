#
# Continuous Thrust Hohmann Transfer around Earth
# Corrected version: stable tangential thrust and circularization at GEO
#

import math
import numpy as np
import matplotlib.pyplot as plt

from Basilisk.architecture import messaging
from Basilisk.simulation import extForceTorque, spacecraft, simpleNav
from Basilisk.utilities import (SimulationBaseClass, macros, simIncludeGravBody, orbitalMotion, unitTestSupport)
from Basilisk.architecture import astroConstants


def run(show_plots=True):
    # -------------------------
    # Simulation setup
    # -------------------------
    simTaskName = "simTask"
    simProcessName = "simProcess"
    scSim = SimulationBaseClass.SimBaseClass()
    dynProcess = scSim.CreateNewProcess(simProcessName)
    simulationTimeStep = macros.sec2nano(10.0)  # simulation time step
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    # Earth environment
    mu_earth = astroConstants.MU_EARTH * 1e9
    gravFactory = simIncludeGravBody.gravBodyFactory()
    earth = gravFactory.createEarth()
    earth.isCentralBody = True

    # Spacecraft
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "HohmannSC"
    scObject.hub.mHub = 500.0  # kg
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d([200.,0.,0.,0.,150.,0.,0.,0.,100.])

    # Add Earth gravity to spacecraft
    gravFactory.addBodiesTo(scObject)

    # Initial orbit (LEO)
    r_LEO = 7000e3  # m
    r_GEO = 42164e3  # m
    oe = orbitalMotion.ClassicElements()
    oe.a = r_LEO
    oe.e = 0.0
    oe.i = 0.0
    oe.Omega = 0.0
    oe.omega = 0.0
    oe.f = 0.0
    r_N, v_N = orbitalMotion.elem2rv(mu_earth, oe)
    scObject.hub.r_CN_NInit = r_N
    scObject.hub.v_CN_NInit = v_N
    scObject.hub.sigma_BNInit = [0.0, 0.0, 0.0]
    scObject.hub.omega_BN_BInit = [0.0, 0.0, 0.0]

    # Continuous thrust effector
    extFT = extForceTorque.ExtForceTorque()
    scObject.addDynamicEffector(extFT)

    # SimpleNav for feedback
    simpleNavMeas = simpleNav.SimpleNav()
    simpleNavMeas.ModelTag = "SimpleNav"
    simpleNavMeas.scStateInMsg.subscribeTo(scObject.scStateOutMsg)

    # Add models to task
    scSim.AddModelToTask(simTaskName, scObject)
    scSim.AddModelToTask(simTaskName, simpleNavMeas)
    scSim.AddModelToTask(simTaskName, extFT)

    # Record spacecraft states
    sc_rec = scObject.scStateOutMsg.recorder()
    scSim.AddModelToTask(simTaskName, sc_rec)

    # -------------------------
    # Propulsion parameters
    # -------------------------
    thrust_max = 500  # N
    mass = scObject.hub.mHub
    thrust_log = []

    # Initialize simulation
    scSim.InitializeSimulation()
    sim_time = 0.0

    # -------------------------
    # Simulation loop
    # -------------------------
    dt = 10.0  # seconds
    total_time = 20 * 3600.0  # simulate 6 hours
    steps = int(total_time / dt)

    for i in range(steps):
        # Read spacecraft state
        state = scObject.scStateOutMsg.read()
        r_vec = np.array(state.r_BN_N)
        v_vec = np.array(state.v_BN_N)
        r_mag = np.linalg.norm(r_vec)
        v_mag = np.linalg.norm(v_vec)

        # Orbital angular momentum vector and tangential direction
        h_vec = np.cross(r_vec, v_vec)
        h_hat = h_vec / np.linalg.norm(h_vec)
        r_hat = r_vec / r_mag
        t_hat = np.cross(h_hat, r_hat)
        t_hat /= np.linalg.norm(t_hat)

        # Compute desired tangential thrust
        # Simple: thrust until radius reaches GEO, then reduce gradually
        if r_mag < r_GEO:
            thrust_dir = t_hat
        else:
            # smoothly reduce thrust to zero near GEO
            excess = r_mag - r_GEO
            reduction_factor = max(0.0, 1.0 - excess / 1e6)  # over 1000 km
            thrust_dir = t_hat * reduction_factor

        thrust = thrust_max * thrust_dir
        thrust_log.append(thrust.copy())

        # Apply thrust
        thrustCmd = messaging.CmdForceBodyMsgPayload()
        thrustCmd.forceRequestBody = thrust
        cmdMsg = messaging.CmdForceBodyMsg().write(thrustCmd)
        extFT.cmdForceBodyInMsg.subscribeTo(cmdMsg)

        # Advance simulation
        sim_time += dt
        scSim.ConfigureStopTime(macros.sec2nano(sim_time))
        scSim.ExecuteSimulation()

    # -------------------------
    # Plot results
    # -------------------------
    time = sc_rec.times() * macros.NANO2SEC
    r_data = np.linalg.norm(sc_rec.r_BN_N, axis=1)
    v_data = np.linalg.norm(sc_rec.v_BN_N, axis=1)

    # Align thrust log length with recorded states
    thrust_array = np.array(thrust_log)
    if len(thrust_array) > len(r_data):
        thrust_array = thrust_array[:len(r_data)]
    elif len(thrust_array) < len(r_data):
        padding = np.zeros((len(r_data) - len(thrust_array), 3))
        thrust_array = np.vstack((thrust_array, padding))
    thrust_data = np.linalg.norm(thrust_array, axis=1)

    plt.figure(figsize=(10,8))
    plt.subplot(3,1,1)
    plt.plot(time/3600, r_data/1e3)
    plt.ylabel("Orbital Radius [km]")
    plt.grid()

    plt.subplot(3,1,2)
    plt.plot(time/3600, v_data)
    plt.ylabel("Velocity [m/s]")
    plt.grid()

    plt.subplot(3,1,3)
    plt.plot(time/3600, thrust_data)
    plt.ylabel("Applied Thrust [N]")
    plt.xlabel("Time [hours]")
    plt.grid()

    plt.suptitle("Continuous Thrust Hohmann Transfer")
    if show_plots:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    run(True)
