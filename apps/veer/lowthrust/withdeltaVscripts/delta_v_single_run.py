# Full simplified single-run simulation with real-time delta-v integration
# (Your original run_single_sim expanded and updated)

import os
import sys
import numpy as np
from Basilisk import __path__
from Basilisk.architecture import messaging
from Basilisk.simulation import spacecraft
from Basilisk.simulation import extForceTorque
from Basilisk.simulation import simpleNav
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport
from Basilisk.utilities import simIncludeGravBody

bskPath = __path__[0]

# ----------------------------------------------------------------------
# UPDATED run_single_sim WITH REAL-TIME Δv ACCUMULATION
# ----------------------------------------------------------------------
def run_single_sim(m_0, T_max, I_sp, a_init=500.0, a_final=35000.0):
    """
    Run a single LEO → GEO transfer
    Includes REAL-TIME Δv integration.
    """

    # --- Constants ---
    R_E = 6378.0
    mu = 3.986e5

    # Initial orbit parameters
    r_init = a_init + R_E
    v_init = np.sqrt(mu / r_init)

    # Final orbit radius
    r_final = a_final + R_E

    # Engine
    T = T_max / 1000.0  # N → kN
    g_0 = 9.807e-3       # gravity in km/s²

    # Simulation parameters
    dynTaskName = "dynTask"
    dynProcessName = "dynProcess"

    scSim = SimulationBaseClass.SimBaseClass()
    dynProcess = scSim.CreateNewProcess(dynProcessName)
    simTimeStep = macros.sec2nano(1.0)
    dynProcess.addTask(scSim.CreateNewTask(dynTaskName, simTimeStep))

    # Spacecraft
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "LEO-GEO-Spacecraft"

    I = [900., 0., 0.,
         0., 800., 0.,
         0., 0., 600.]

    scObject.hub.mHub = m_0
    scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)

    scObject.hub.r_CN_NInit = [[r_init * 1000.0], [0.0], [0.0]]
    scObject.hub.v_CN_NInit = [[0.0], [v_init * 1000.0], [0.0]]
    scObject.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]
    scObject.hub.omega_BN_BInit = [[0.0], [0.0], [0.0]]

    scSim.AddModelToTask(dynTaskName, scObject)

    # Gravity
    gravFactory = simIncludeGravBody.gravBodyFactory()
    earth = gravFactory.createEarth()
    earth.isCentralBody = True
    earth.mu = mu * 1e9
    gravFactory.addBodiesTo(scObject)

    # External force/thrust
    extFTObject = extForceTorque.ExtForceTorque()
    extFTObject.ModelTag = "tangentialThrust"
    scObject.addDynamicEffector(extFTObject)
    scSim.AddModelToTask(dynTaskName, extFTObject)

    # Navigation
    sNavObject = simpleNav.SimpleNav()
    sNavObject.ModelTag = "SimpleNavigation"
    scSim.AddModelToTask(dynTaskName, sNavObject)
    sNavObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)

    # Logging
    samplingTime = macros.sec2nano(60.0)
    scStateLog = scObject.scStateOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(dynTaskName, scStateLog)

    scSim.InitializeSimulation()

    # -------------------------------------------------
    # Δv ACCUMULATOR
    # -------------------------------------------------
    delta_v = 0.0  # km/s accumulated in real time

    SAFE_TIME_LIMIT = macros.day2nano(90.0)
    max_simulation_time = macros.day2nano(500.0)
    update_interval = macros.sec2nano(10.0)

    thrust_phase_complete = False
    orbit_count = 0
    last_y = 0.0

    thrust_positions = []
    thrust_velocities = []
    thrust_times = []

    total_elapsed_time = 0.0
    segment_count = 0

    current_scSim = scSim
    current_scObject = scObject
    current_extFTObject = extFTObject

    # ======================================================
    # MAIN THRUST LOOP
    # ======================================================
    while total_elapsed_time < (max_simulation_time * macros.NANO2SEC) and not thrust_phase_complete:
        segment_count += 1
        segment_max_time = SAFE_TIME_LIMIT
        current_time = 0

        while current_time < segment_max_time and not thrust_phase_complete:
            # Step simulation
            current_scSim.ConfigureStopTime(current_time + update_interval)
            current_scSim.ExecuteSimulation()

            # State
            r_BN_N = current_scObject.dynManager.getStateObject("hubPosition").getState()
            v_BN_N = current_scObject.dynManager.getStateObject("hubVelocity").getState()
            current_mass = current_scObject.hub.mHub

            r_vec = np.array([r_BN_N[0][0], r_BN_N[1][0], r_BN_N[2][0]]) / 1000.0
            v_vec = np.array([v_BN_N[0][0], v_BN_N[1][0], v_BN_N[2][0]]) / 1000.0

            r_mag = np.linalg.norm(r_vec)
            v_mag = np.linalg.norm(v_vec)

            # Store
            thrust_positions.append(r_vec.copy())
            thrust_velocities.append(v_vec.copy())
            thrust_times.append(total_elapsed_time + current_time * macros.NANO2SEC)

            # Orbit count
            current_y = r_vec[1]
            if last_y < 0 and current_y >= 0:
                orbit_count += 1
            last_y = current_y

            if r_mag >= r_final:
                thrust_phase_complete = True
                break

            # ---------------------------------------------
            # Tangential thrust direction
            # ---------------------------------------------
            if v_mag > 0:
                thrust_dir = v_vec / v_mag
                thrust_force_N = (T * 1000.0) * thrust_dir
                current_extFTObject.extForce_N = [[thrust_force_N[0]],
                                                  [thrust_force_N[1]],
                                                  [thrust_force_N[2]]]
            else:
                current_extFTObject.extForce_N = [[0.0], [0.0], [0.0]]

            # ---------------------------------------------
            # Δv INTEGRATION IN REAL TIME
            # ---------------------------------------------
            dt = update_interval * macros.NANO2SEC
            accel = T / current_mass  # km/s²
            delta_v += accel * dt     # km/s

            # ---------------------------------------------
            # MASS UPDATE
            # ---------------------------------------------
            dm = -T / (I_sp * g_0) * dt
            current_scObject.hub.mHub = max(10.0, current_mass + dm)

            current_time += update_interval

        total_elapsed_time += current_time * macros.NANO2SEC

        # Break if done
        if thrust_phase_complete:
            break

        # If more segments needed, rebuild simulation
        if total_elapsed_time < (max_simulation_time * macros.NANO2SEC):
            r_state = current_scObject.dynManager.getStateObject("hubPosition").getState()
            v_state = current_scObject.dynManager.getStateObject("hubVelocity").getState()
            m_state = current_scObject.hub.mHub

            current_scSim = SimulationBaseClass.SimBaseClass()
            dynProcess_new = current_scSim.CreateNewProcess(dynProcessName + f"_{segment_count}")
            dynProcess_new.addTask(current_scSim.CreateNewTask(dynTaskName + f"_{segment_count}", simTimeStep))

            current_scObject = spacecraft.Spacecraft()
            current_scObject.ModelTag = f"Seg{segment_count}"
            current_scObject.hub.mHub = m_state
            current_scObject.hub.r_CN_NInit = r_state
            current_scObject.hub.v_CN_NInit = v_state
            current_scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)
            current_scObject.hub.r_BcB_B = [[0.0],[0.0],[0.0]]
            current_scObject.hub.sigma_BNInit = [[0.0],[0.0],[0.0]]
            current_scObject.hub.omega_BN_BInit = [[0.0],[0.0],[0.0]]

            current_scSim.AddModelToTask(dynTaskName + f"_{segment_count}", current_scObject)

            gravFactory_new = simIncludeGravBody.gravBodyFactory()
            earth_new = gravFactory_new.createEarth()
            earth_new.isCentralBody = True
            earth_new.mu = mu * 1e9
            gravFactory_new.addBodiesTo(current_scObject)

            current_extFTObject = extForceTorque.ExtForceTorque()
            current_extFTObject.ModelTag = f"thrust_{segment_count}"
            current_scObject.addDynamicEffector(current_extFTObject)
            current_scSim.AddModelToTask(dynTaskName + f"_{segment_count}", current_extFTObject)

            current_scSim.InitializeSimulation()

    final_mass = current_scObject.hub.mHub
    prop_used = m_0 - final_mass

    # -------------------------------------------------------------
    # RETURN RESULTS (WITH Δv)
    # -------------------------------------------------------------
    results = {
        'Initial_Altitude_km': a_init,
        'Target_Altitude_km': a_final,
        'Initial_Mass_kg': m_0,
        'Final_Mass_kg': final_mass,
        'Propellant_Used_kg': prop_used,
        'Time_of_Flight_days': total_elapsed_time / 86400.0,
        'Number_of_Orbits': orbit_count,
        'Number_of_Segments': segment_count,
        'DeltaV_km_s': delta_v
    }

    return results


# --------------------------------------------------------------
# MAIN — SINGLE RUN ONLY
# --------------------------------------------------------------
def main():
    print("=" * 60)
    print(" SINGLE LEO → GEO SIMULATION (Δv ENABLED) ")
    print("=" * 60)

    m_0 = 1000.0
    T_max = 1.0
    I_sp = 10000.0

    print(f"Mass:   {m_0} kg")
    print(f"Thrust: {T_max} N")
    print(f"Isp:    {I_sp} s")

    results = run_single_sim(m_0, T_max, I_sp)

    print("\nSimulation Results:")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()