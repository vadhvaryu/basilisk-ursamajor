"""
LEO to GEO Low Thrust Orbit Transfer using Basilisk

This script simulates a spacecraft performing a low-thrust spiral transfer
from Low Earth Orbit (LEO) to Geostationary Earth Orbit (GEO) using continuous
tangential thrust with Basilisk's astrodynamics framework.

Parametric study version - iterates over mass, thrust, and Isp ranges.

Author: Based on Basilisk examples
"""

import os
import sys
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

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

def run_single_sim(m_0, T_max, I_sp, a_init=500.0, a_final=35000.0):
    """
    Run a single LEO to GEO transfer simulation
    
    Args:
        m_0 (float): Initial spacecraft mass [kg]
        T_max (float): Maximum thrust [N]
        I_sp (float): Specific impulse [s]
        a_init (float): Initial altitude [km]
        a_final (float): Final altitude [km]
        
    Returns:
        dict: Results dictionary with all metrics
    """
    
    # --- Constants ---
    R_E = 6378.0  # km, Earth radius
    mu = 3.986e5  # km^3/s^2, Earth gravitational parameter
    
    # Initial orbit parameters
    r_init = a_init + R_E  # km, orbital radius
    v_init = np.sqrt(mu / r_init)  # km/s, circular orbit velocity
    
    # Final orbit parameters
    r_final = a_final + R_E  # km
    
    # Engine parameters
    T = T_max / 1000.0  # Convert N to kN
    g_0 = 9.807e-3  # km/s^2, standard gravity
    
    # --- Simulation Parameters ---
    dynTaskName = "dynTask"
    dynProcessName = "dynProcess"
    
    # Create simulation
    scSim = SimulationBaseClass.SimBaseClass()
    
    # Create process and task
    dynProcess = scSim.CreateNewProcess(dynProcessName)
    simTimeStep = macros.sec2nano(1.0)  # 1 second time steps
    dynProcess.addTask(scSim.CreateNewTask(dynTaskName, simTimeStep))
    
    # --- Spacecraft Setup ---
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "LEO-GEO-Spacecraft"
    
    # Mass properties
    I = [900., 0., 0.,
         0., 800., 0.,
         0., 0., 600.]
    scObject.hub.mHub = m_0
    scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)
    
    # Initial conditions - circular LEO orbit in XY plane
    scObject.hub.r_CN_NInit = [[r_init * 1000.0], [0.0], [0.0]]  # m
    scObject.hub.v_CN_NInit = [[0.0], [v_init * 1000.0], [0.0]]  # m/s
    scObject.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]
    scObject.hub.omega_BN_BInit = [[0.0], [0.0], [0.0]]  # rad/s
    
    scSim.AddModelToTask(dynTaskName, scObject)
    
    # --- Gravity Setup ---
    gravFactory = simIncludeGravBody.gravBodyFactory()
    earth = gravFactory.createEarth()
    earth.isCentralBody = True
    earth.mu = mu * 1e9  # Convert km^3/s^2 to m^3/s^2
    gravFactory.addBodiesTo(scObject)
    
    # --- External Force for Tangential Thrust ---
    extFTObject = extForceTorque.ExtForceTorque()
    extFTObject.ModelTag = "tangentialThrust"
    scObject.addDynamicEffector(extFTObject)
    scSim.AddModelToTask(dynTaskName, extFTObject)
    
    # --- Navigation ---
    sNavObject = simpleNav.SimpleNav()
    sNavObject.ModelTag = "SimpleNavigation"
    scSim.AddModelToTask(dynTaskName, sNavObject)
    sNavObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
    
    # --- Data Logging ---
    samplingTime = macros.sec2nano(60.0)  # Log every 60 seconds
    scStateLog = scObject.scStateOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(dynTaskName, scStateLog)
    
    # --- Initialize Simulation ---
    scSim.InitializeSimulation()
    
    # --- THRUST PHASE ---
    SAFE_TIME_LIMIT = macros.day2nano(90.0)  # 90 days per segment
    max_simulation_time = macros.day2nano(500.0)  # 500 days total max
    update_interval = macros.sec2nano(10.0)  # Update thrust every 10 seconds
    
    thrust_phase_complete = False
    orbit_count = 0
    last_y = 0.0
    
    # Storage for trajectory
    thrust_positions = []
    thrust_velocities = []
    thrust_times = []
    
    # Track total elapsed time across all segments
    total_elapsed_time = 0.0  # in seconds
    segment_count = 0
    
    # Initial state
    current_scObject = scObject
    current_scSim = scSim
    current_extFTObject = extFTObject

    ### Δv addition ###
    delta_v = 0.0  # km/s accumulated from thrust acceleration
    
    while total_elapsed_time < (max_simulation_time * macros.NANO2SEC) and not thrust_phase_complete:
        segment_count += 1
        
        # Run this segment up to safe time limit
        segment_max_time = SAFE_TIME_LIMIT
        current_time = 0
        
        while current_time < segment_max_time and not thrust_phase_complete:
            # Execute simulation step
            current_scSim.ConfigureStopTime(current_time + update_interval)
            current_scSim.ExecuteSimulation()
            
            # Get current state (in meters and m/s)
            r_BN_N = current_scObject.dynManager.getStateObject("hubPosition").getState()
            v_BN_N = current_scObject.dynManager.getStateObject("hubVelocity").getState()
            current_mass = current_scObject.hub.mHub
            
            # Convert to km and km/s for calculations
            r_vec = np.array([r_BN_N[0][0], r_BN_N[1][0], r_BN_N[2][0]]) / 1000.0
            v_vec = np.array([v_BN_N[0][0], v_BN_N[1][0], v_BN_N[2][0]]) / 1000.0
            
            r_mag = np.linalg.norm(r_vec)
            v_mag = np.linalg.norm(v_vec)
            
            # Store trajectory data
            thrust_positions.append(r_vec.copy())
            thrust_velocities.append(v_vec.copy())
            thrust_times.append(total_elapsed_time + current_time * macros.NANO2SEC)
            
            # Check for orbit crossing (Y coordinate sign change)
            current_y = r_vec[1]
            if last_y < 0 and current_y >= 0 and len(thrust_times) > 1:
                orbit_count += 1
            last_y = current_y
            
            # Check if reached destination
            if r_mag >= r_final:
                thrust_phase_complete = True
                break
            
            # Calculate tangential thrust force
            if v_mag > 0:
                thrust_dir = v_vec / v_mag
                thrust_force_kN = T * thrust_dir  # kN
                thrust_force_N = thrust_force_kN * 1000.0  # N
                
                current_extFTObject.extForce_N = [
                    [thrust_force_N[0]],
                    [thrust_force_N[1]],
                    [thrust_force_N[2]]
                ]
            else:
                current_extFTObject.extForce_N = [[0.0], [0.0], [0.0]]
            
            # Update mass (fuel consumption)
            dt = update_interval * macros.NANO2SEC  # seconds
            dm = -T / (I_sp * g_0) * dt  # kg
            current_scObject.hub.mHub = max(10.0, current_mass + dm)

            ### Δv addition (real-time integration) ###
            accel = T / current_mass            # km/s^2  (T is in kN, consistent with your logic)
            delta_v += accel * dt               # km/s    (dt is seconds)
            
            current_time += update_interval
        
        # Update total elapsed time
        total_elapsed_time += current_time * macros.NANO2SEC
        
        # If not complete, create new simulation for next segment
        if not thrust_phase_complete and total_elapsed_time < (max_simulation_time * macros.NANO2SEC):
            # Save current state
            r_BN_N_seg = current_scObject.dynManager.getStateObject("hubPosition").getState()
            v_BN_N_seg = current_scObject.dynManager.getStateObject("hubVelocity").getState()
            current_mass_seg = current_scObject.hub.mHub
            
            # Create new simulation
            current_scSim = SimulationBaseClass.SimBaseClass()
            dynProcess_new = current_scSim.CreateNewProcess(dynProcessName + f"_{segment_count}")
            dynProcess_new.addTask(current_scSim.CreateNewTask(dynTaskName + f"_{segment_count}", simTimeStep))
            
            # Create new spacecraft
            current_scObject = spacecraft.Spacecraft()
            current_scObject.ModelTag = f"LEO-GEO-Spacecraft-Seg{segment_count}"
            current_scObject.hub.mHub = current_mass_seg
            current_scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
            current_scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)
            current_scObject.hub.r_CN_NInit = r_BN_N_seg
            current_scObject.hub.v_CN_NInit = v_BN_N_seg
            current_scObject.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]
            current_scObject.hub.omega_BN_BInit = [[0.0], [0.0], [0.0]]
            
            current_scSim.AddModelToTask(dynTaskName + f"_{segment_count}", current_scObject)
            
            # Add gravity
            gravFactory_new = simIncludeGravBody.gravBodyFactory()
            earth_new = gravFactory_new.createEarth()
            earth_new.isCentralBody = True
            earth_new.mu = mu * 1e9
            gravFactory_new.addBodiesTo(current_scObject)
            
            # Add external force
            current_extFTObject = extForceTorque.ExtForceTorque()
            current_extFTObject.ModelTag = f"tangentialThrust_{segment_count}"
            current_scObject.addDynamicEffector(current_extFTObject)
            current_scSim.AddModelToTask(dynTaskName + f"_{segment_count}", current_extFTObject)
            
            # Initialize new simulation
            current_scSim.InitializeSimulation()
    
    thrust_end_time_sec = total_elapsed_time
    final_thrust_mass = current_scObject.hub.mHub
    propellant_used = m_0 - final_thrust_mass
    
    # --- COAST PHASE ---
    r_BN_N_final = current_scObject.dynManager.getStateObject("hubPosition").getState()
    v_BN_N_final = current_scObject.dynManager.getStateObject("hubVelocity").getState()
    
    # Create a NEW simulation for coast phase
    scSim2 = SimulationBaseClass.SimBaseClass()
    dynProcess2 = scSim2.CreateNewProcess(dynProcessName + "_coast")
    dynProcess2.addTask(scSim2.CreateNewTask(dynTaskName + "_coast", simTimeStep))
    
    # Create new spacecraft
    scObject2 = spacecraft.Spacecraft()
    scObject2.ModelTag = "LEO-GEO-Spacecraft-Coast"
    scObject2.hub.mHub = final_thrust_mass
    scObject2.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
    scObject2.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)
    scObject2.hub.r_CN_NInit = r_BN_N_final
    scObject2.hub.v_CN_NInit = v_BN_N_final
    scObject2.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]
    scObject2.hub.omega_BN_BInit = [[0.0], [0.0], [0.0]]
    
    scSim2.AddModelToTask(dynTaskName + "_coast", scObject2)
    
    # Add gravity
    gravFactory2 = simIncludeGravBody.gravBodyFactory()
    earth2 = gravFactory2.createEarth()
    earth2.isCentralBody = True
    earth2.mu = mu * 1e9
    gravFactory2.addBodiesTo(scObject2)
    
    # Add navigation
    sNavObject2 = simpleNav.SimpleNav()
    sNavObject2.ModelTag = "SimpleNavigation2"
    scSim2.AddModelToTask(dynTaskName + "_coast", sNavObject2)
    sNavObject2.scStateInMsg.subscribeTo(scObject2.scStateOutMsg)
    
    # Data logging
    coast_sample_interval = macros.sec2nano(600.0)
    scStateLog2 = scObject2.scStateOutMsg.recorder(coast_sample_interval)
    scSim2.AddModelToTask(dynTaskName + "_coast", scStateLog2)
    
    # Initialize and run
    scSim2.InitializeSimulation()
    t_coast = macros.day2nano(14.0)
    scSim2.ConfigureStopTime(t_coast)
    scSim2.ExecuteSimulation()
    
    # Extract coast phase data
    coast_r_BN_N = scStateLog2.r_BN_N / 1000.0  # Convert to km
    coast_v_BN_N = scStateLog2.v_BN_N / 1000.0  # Convert to km/s
    
    coast_positions = [coast_r_BN_N[i] for i in range(len(coast_r_BN_N))]
    
    # Analyze coast phase orbit
    coast_radii = [np.linalg.norm(r) for r in coast_positions]
    r_coast_apo = max(coast_radii)
    r_coast_per = min(coast_radii)
    r_coast_avg = (r_coast_apo + r_coast_per) / 2.0
    r_coast_error = (r_coast_avg - r_final) / r_final * 100.0
    e_coast = (r_coast_apo - r_coast_per) / (r_coast_apo + r_coast_per)
    
    final_v_vec = coast_v_BN_N[-1]
    v_final_coast = np.linalg.norm(final_v_vec)
    
    # Return results dictionary
    results = {
        'Initial_Altitude_km': a_init,
        'Target_Altitude_km': a_final,
        'Initial_Mass_kg': m_0,
        'Max_Thrust_N': T_max,
        'Specific_Impulse_s': I_sp,
        'Initial_Speed_km_s': v_init,
        'Final_Speed_Thrust_km_s': v_mag,
        'Final_Mass_kg': final_thrust_mass,
        'Propellant_Used_kg': propellant_used,
        'Time_of_Flight_days': thrust_end_time_sec / 86400.0,
        'Number_of_Orbits': orbit_count,
        'Number_of_Segments': segment_count,
        'Coast_Apoapsis_km': r_coast_apo,
        'Coast_Periapsis_km': r_coast_per,
        'Coast_Radius_Error_percent': r_coast_error,
        'Coast_Eccentricity': e_coast,
        'Speed_After_Coast_km_s': v_final_coast,
        'DeltaV_km_s': delta_v
    }
    
    return results


def run_parametric_study(show_plots=False, save_csv=True):
    """
    Run parametric study over mass and thrust ranges
    """
    
    # Suppress duplicate output
    sys.stdout.reconfigure(line_buffering=True)
    
    # Define parameter ranges
    m_0_range = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0]
    T_max_range = [0.5, 1, 2, 3, 4, 5, 7, 9] 
    I_sp = 10000.0  # Fixed specific impulse
    
    # Calculate total number of runs
    total_runs = len(m_0_range) * len(T_max_range)
    current_run = 0
    
    print("=" * 60)
    print("LEO to GEO PARAMETRIC STUDY")
    print("=" * 60)
    print(f"Total simulation runs: {total_runs}")
    print(f"Mass range: {m_0_range} kg")
    print(f"Thrust range: {T_max_range} N")
    print(f"Specific Impulse (fixed): {I_sp} s")
    print("=" * 60)
    print("Starting simulations...")
    
    # Store all results
    all_results = []
    
    # Iterate over all parameter combinations
    for m_0 in m_0_range:
        for T_max in T_max_range:
            current_run += 1
            
            print(f"Run {current_run}/{total_runs}: m={m_0}kg, T={T_max}N", flush=True)
            
            try:
                # Run simulation
                results = run_single_sim(m_0, T_max, I_sp)
                all_results.append(results)
                
                print(f"  ✓ Completed in {results['Time_of_Flight_days']:.2f} days, "
                      f"{results['Propellant_Used_kg']:.2f} kg propellant", flush=True)
                
            except Exception as e:
                print(f"  ✗ FAILED: {str(e)}", flush=True)
                # Add failure record
                all_results.append({
                    'Initial_Mass_kg': m_0,
                    'Max_Thrust_N': T_max,
                    'Specific_Impulse_s': I_sp,
                    'Status': 'FAILED',
                    'Error': str(e)
                })
    
    # Save all results to CSV
    if save_csv and len(all_results) > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"leo_geo_parametric_study_{timestamp}.csv"
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        df.to_csv(csv_filename, index=False)
        
        print("" + "=" * 60)
        print("PARAMETRIC STUDY COMPLETE")
        print("=" * 60)
        print(f"Total runs completed: {current_run}")
        print(f"Results saved to: {csv_filename}")
        print("=" * 60)
    
    return all_results


if __name__ == "__main__":
    run_parametric_study(show_plots=False, save_csv=True)
