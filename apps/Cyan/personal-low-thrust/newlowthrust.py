"""
LEO to GEO Low Thrust Orbit Transfer using Basilisk

Converted from scipy-based implementation (lowthrust.py) to Basilisk framework.
Simulates continuous tangential thrust from LEO (500 km) to GEO (35,000 km)
followed by a 2-week coast phase to verify orbit stability.

All units in SI: meters, m/s, kg, seconds, Newtons
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport
from Basilisk.utilities import simIncludeGravBody
from Basilisk.simulation import spacecraft
from Basilisk.simulation import extForceTorque
from Basilisk.simulation import simpleNav
from Basilisk import __path__

bskPath = __path__[0]


def run(show_plots=True):
    """
    Execute LEO to GEO low-thrust transfer 
    
    Args:
        show_plots (bool): Display trajectory and performance plots
        
    Returns:
        scSim: Basilisk simulation object
    """
    
    # ========================================================================
    # SIMULATION CONSTANTS (SI UNITS)
    
    # Earth parameters
    R_E = 6.378e6  # m, Earth radius
    mu = 3.986004418e14  # m^3/s^2, gravitational parameter
    
    # Orbit parameters
    a_init = 500.0e3  # m, initial altitude (500 km)
    r_init = R_E + a_init  # m, initial orbital radius
    v_init = np.sqrt(mu / r_init)  # m/s, circular orbit velocity
    
    a_final = 35000.0e3  # m, final altitude (35,000 km)
    r_final = R_E + a_final  # m, target orbital radius
    
    # Spacecraft parameters
    m_0 = 1000.0  # kg, initial spacecraft mass
    
    # Propulsion system parameters
    T_max = 1.0  # N, maximum thrust
    I_sp = 10000.0  # s, specific impulse
    g_0 = 9.80665  # m/s^2, standard gravity
    
    # Print simulation parameters
    print("="*70)
    print("Low-Thrust Orbit Transfer: LEO → GEO")
    print("="*70)
    print(f"Initial Altitude:    {a_init/1e3:.1f} km")
    print(f"Target Altitude:     {a_final/1e3:.1f} km")
    print(f"Thrust:              {T_max:.1f} N")
    print(f"Specific Impulse:    {I_sp:.0f} s")
    print(f"Initial Mass:        {m_0:.1f} kg")
    print(f"Initial Velocity:    {v_init:.2f} m/s")
    print("="*70)
    
    # ========================================================================
    # CREATE BASILISK SIMULATION ENVIRONMENT
    
    scSim = SimulationBaseClass.SimBaseClass()
    
    # Create dynamics process and task
    dynProcess = scSim.CreateNewProcess("dynamicsProcess")
    simTimeStep = macros.sec2nano(1.0)  # 1 second time step
    dynProcess.addTask(scSim.CreateNewTask("dynamicsTask", simTimeStep))
    
    # ========================================================================
    # SPACECRAFT CONFIGURATION
    
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "spacecraft"
    
    # Set mass properties
    scObject.hub.mHub = m_0
    scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
    
    # Moment of inertia tensor
    I_tensor = [900., 0., 0.,
                0., 800., 0.,
                0., 0., 600.]
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I_tensor)
    
    # Initial state: circular LEO orbit in XY plane
    scObject.hub.r_CN_NInit = [[r_init], [0.0], [0.0]]  # m, position
    scObject.hub.v_CN_NInit = [[0.0], [v_init], [0.0]]  # m/s, velocity
    scObject.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]  # attitude
    scObject.hub.omega_BN_BInit = [[0.0], [0.0], [0.0]]  # angular velocity
    
    scSim.AddModelToTask("dynamicsTask", scObject)
    
    # ========================================================================
    # GRAVITY MODEL
    
    gravFactory = simIncludeGravBody.gravBodyFactory()
    earth = gravFactory.createEarth()
    earth.isCentralBody = True
    earth.mu = mu  # Set exact gravitational parameter
    gravFactory.addBodiesTo(scObject)
    
    # ========================================================================
    # THRUST SYSTEM (External Force)
    
    extForce = extForceTorque.ExtForceTorque()
    extForce.ModelTag = "thrustForce"
    scObject.addDynamicEffector(extForce)
    scSim.AddModelToTask("dynamicsTask", extForce)
    
    # ========================================================================
    # NAVIGATION
    
    navObject = simpleNav.SimpleNav()
    navObject.ModelTag = "navigation"
    navObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
    scSim.AddModelToTask("dynamicsTask", navObject)
    
    # ========================================================================
    # DATA RECORDING
    
    dataLog = scObject.scStateOutMsg.recorder(simTimeStep)
    scSim.AddModelToTask("dynamicsTask", dataLog)
    
    # ========================================================================
    # INITIALIZE SIMULATION
    
    scSim.InitializeSimulation()
    
    # ========================================================================
    # THRUST PHASE SIMULATION
    
    print("\n>>> Starting Thrust Phase...")
    
    sim_time = 0.0  # seconds
    max_sim_time = 200.0 * 86400.0  # 200 days in seconds
    control_update = 10.0  # seconds, thrust update interval
    
    # State tracking
    thrust_active = True
    orbit_crossings = 0
    previous_y = 0.0
    
    # Data storage
    thrust_r_history = []
    thrust_v_history = []
    thrust_t_history = []
    thrust_m_history = []
    
    while sim_time < max_sim_time and thrust_active:
        
        # Run simulation step
        scSim.ConfigureStopTime(macros.sec2nano(sim_time + control_update))
        scSim.ExecuteSimulation()
        
        # Extract current state
        r_BN = scObject.dynManager.getStateObject("hubPosition").getState()
        v_BN = scObject.dynManager.getStateObject("hubVelocity").getState()
        mass = scObject.hub.mHub
        
        # Convert to numpy arrays
        r_vec = np.array([r_BN[0][0], r_BN[1][0], r_BN[2][0]])  # m
        v_vec = np.array([v_BN[0][0], v_BN[1][0], v_BN[2][0]])  # m/s
        
        r_mag = np.linalg.norm(r_vec)
        v_mag = np.linalg.norm(v_vec)
        
        # Record data
        thrust_r_history.append(r_vec.copy())
        thrust_v_history.append(v_vec.copy())
        thrust_t_history.append(sim_time)
        thrust_m_history.append(mass)
        
        # Count orbit crossings (Y-axis crossings from below)
        if previous_y < 0 and r_vec[1] >= 0 and sim_time > 0:
            orbit_crossings += 1
        previous_y = r_vec[1]
        
        # Check termination condition: reached target radius
        if r_mag >= r_final:
            thrust_active = False
            print(f"\n✓ Target altitude reached!")
            print(f"  Time: {sim_time/86400.0:.4f} days")
            break
        
        # Apply tangential thrust (in velocity direction)
        if v_mag > 0:
            thrust_direction = v_vec / v_mag
            thrust_force = T_max * thrust_direction  # N
            
            extForce.extForce_N = [[thrust_force[0]], 
                                   [thrust_force[1]], 
                                   [thrust_force[2]]]
        else:
            extForce.extForce_N = [[0.0], [0.0], [0.0]]
        
        # Update mass: dm/dt = -T / (Isp * g0)
        delta_t = control_update  # seconds
        delta_m = -T_max / (I_sp * g_0) * delta_t  # kg
        scObject.hub.mHub = max(10.0, mass + delta_m)
        
        sim_time += control_update
    
    thrust_end_time = sim_time
    final_mass_thrust = scObject.hub.mHub
    fuel_used = m_0 - final_mass_thrust
    
    # Final thrust phase velocity
    v_BN = scObject.dynManager.getStateObject("hubVelocity").getState()
    v_final_thrust = np.linalg.norm([v_BN[0][0], v_BN[1][0], v_BN[2][0]])
    
    print(f"\nThrust Phase Results:")
    print(f"  Duration:        {thrust_end_time/86400.0:.4f} days")
    print(f"  Orbits:          {orbit_crossings}")
    print(f"  Fuel consumed:   {fuel_used:.4f} kg")
    print(f"  Final velocity:  {v_final_thrust:.3f} m/s")
    
    # ========================================================================
    # COAST PHASE SIMULATION (2 weeks)
    
    print("\n>>> Starting Coast Phase (14 days)...")
    
    # Disable thrust
    extForce.extForce_N = [[0.0], [0.0], [0.0]]
    
    coast_duration = 14.0 * 86400.0  # 2 weeks in seconds
    coast_end_time = thrust_end_time + coast_duration
    coast_sample_rate = 600.0  # seconds, sample every 10 minutes
    
    coast_r_history = []
    coast_v_history = []
    coast_t_history = []
    
    while sim_time < coast_end_time:
        next_time = min(sim_time + coast_sample_rate, coast_end_time)
        
        scSim.ConfigureStopTime(macros.sec2nano(next_time))
        scSim.ExecuteSimulation()
        
        # Extract state
        r_BN = scObject.dynManager.getStateObject("hubPosition").getState()
        v_BN = scObject.dynManager.getStateObject("hubVelocity").getState()
        
        r_vec = np.array([r_BN[0][0], r_BN[1][0], r_BN[2][0]])
        v_vec = np.array([v_BN[0][0], v_BN[1][0], v_BN[2][0]])
        
        coast_r_history.append(r_vec.copy())
        coast_v_history.append(v_vec.copy())
        coast_t_history.append(sim_time)
        
        sim_time = next_time
    
    # Analyze coast orbit
    coast_radii = [np.linalg.norm(r) for r in coast_r_history]
    r_apogee = max(coast_radii)
    r_perigee = min(coast_radii)
    r_mean = (r_apogee + r_perigee) / 2.0
    eccentricity = (r_apogee - r_perigee) / (r_apogee + r_perigee)
    radius_error_pct = (r_mean - r_final) / r_final * 100.0
    
    v_final_coast = np.linalg.norm(coast_v_history[-1])
    
    print(f"\nCoast Phase Results:")
    print(f"  Mean radius error:  {radius_error_pct:.3f}%")
    print(f"  Eccentricity:       {eccentricity:.5f}")
    print(f"  Final velocity:     {v_final_coast:.3f} m/s")
    
    # ========================================================================
    # FINAL SUMMARY
    
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    print(f"Initial Altitude:        {a_init/1e3:.1f} km")
    print(f"Target Altitude:         {a_final/1e3:.1f} km")
    print(f"Thrust:                  {T_max:.1f} N")
    print(f"Initial Velocity:        {v_init:.3f} m/s")
    print(f"Final Velocity (thrust): {v_final_thrust:.3f} m/s")
    print(f"Fuel Used:               {fuel_used:.4f} kg")
    print(f"Transfer Time:           {thrust_end_time/86400.0:.4f} days")
    print(f"Number of Orbits:        {orbit_crossings}")
    print(f"Coast Radius Error:      {radius_error_pct:.3f}%")
    print(f"Coast Eccentricity:      {eccentricity:.5f}")
    print(f"Final Velocity (coast):  {v_final_coast:.3f} m/s")
    print("="*70)
    
    # ========================================================================
    # VISUALIZATION
    
    if show_plots:
        
        # Convert lists to arrays
        thrust_r_history = np.array(thrust_r_history)
        coast_r_history = np.array(coast_r_history)
        thrust_v_history = np.array(thrust_v_history)
        coast_v_history = np.array(coast_v_history)
        thrust_t_history = np.array(thrust_t_history)
        coast_t_history = np.array(coast_t_history)
        
        # Summary text for plot
        summary_text = (
            f"Initial Altitude: {a_init/1e3:.1f} km\n"
            f"Final Target Altitude: {a_final/1e3:.1f} km\n"
            f"Max Thrust: {T_max:.3f} N\n"
            f"Initial speed: {v_init:.3f} m/s\n"
            f"Final (thrust cutoff) speed: {v_final_thrust:.3f} m/s\n"
            f"Propellant used: {fuel_used:.4f} kg\n"
            f"Time of flight: {thrust_end_time/86400.0:.4f} days\n"
            f"Number of orbits: {orbit_crossings}\n"
            f"Coast Radius Error: {radius_error_pct:.3f} %\n"
            f"Coast eccentricity: {eccentricity:.5f}\n"
            f"Speed after coast: {v_final_coast:.3f} m/s"
        )
        
        # --- PLOT 1: Trajectory ---
        plt.rc("font", size=18)
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_aspect("equal")
        ax.axis("off")
        
        # Earth
        earth_patch = Circle((0, 0), R_E/1e3, ec="none", fc="C0")
        ax.add_patch(earth_patch)
        ax.annotate("Earth", xy=(0, 0), ha="center", va="center", color="white")
        
        # Target orbit circle
        target_patch = Circle((0, 0), r_final/1e3, ec="C1", fc="none", lw=2, ls="--")
        ax.add_patch(target_patch)
        
        # Trajectories (convert to km for display)
        ax.plot(thrust_r_history[:, 0]/1e3, thrust_r_history[:, 1]/1e3,
                color="C2", lw=1, label="Thrust Phase")
        ax.plot(coast_r_history[:, 0]/1e3, coast_r_history[:, 1]/1e3,
                color="C3", lw=1, label="Coast Phase")
        
        ax.legend(loc="upper right", fontsize=10)
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        # --- PLOT 2: Altitude vs Time ---
        thrust_alt = [np.linalg.norm(r) - R_E for r in thrust_r_history]
        coast_alt = [np.linalg.norm(r) - R_E for r in coast_r_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(thrust_t_history/86400.0, np.array(thrust_alt)/1e3, 
                'g-', linewidth=2, label='Thrust Phase')
        plt.plot(coast_t_history/86400.0, np.array(coast_alt)/1e3, 
                'r-', linewidth=2, label='Coast Phase')
        plt.axhline(y=a_final/1e3, color='b', linestyle='--', 
                   linewidth=1, label='Target Altitude')
        plt.xlabel('Time (days)', fontsize=12)
        plt.ylabel('Altitude (km)', fontsize=12)
        plt.title('Altitude vs Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # --- PLOT 3: Velocity vs Time ---
        thrust_vel_mag = [np.linalg.norm(v) for v in thrust_v_history]
        coast_vel_mag = [np.linalg.norm(v) for v in coast_v_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(thrust_t_history/86400.0, thrust_vel_mag, 
                'g-', linewidth=2, label='Thrust Phase')
        plt.plot(coast_t_history/86400.0, coast_vel_mag, 
                'r-', linewidth=2, label='Coast Phase')
        plt.xlabel('Time (days)', fontsize=12)
        plt.ylabel('Velocity (m/s)', fontsize=12)
        plt.title('Velocity Magnitude vs Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        plt.show()
    
    return scSim


if __name__ == "__main__":
    run(show_plots=True)
