"""
LEO to GEO Low Thrust Orbit Transfer using Basilisk
Python 3.12 compatible
"""

print("="*70)
print("SCRIPT STARTING...")
print("="*70)

import sys
import os
import numpy as np

print("✓ Basic imports successful")

# Import matplotlib
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    print("✓ Matplotlib imported")
except ImportError as e:
    print(f"✗ Matplotlib import failed: {e}")
    sys.exit(1)

# Import Basilisk - it's in your site-packages
try:
    print("\nAttempting to import Basilisk...")
    from Basilisk import __path__ as bsk_path
    from Basilisk.utilities import SimulationBaseClass
    from Basilisk.utilities import macros
    from Basilisk.utilities import unitTestSupport
    from Basilisk.utilities import simIncludeGravBody
    from Basilisk.simulation import spacecraft
    from Basilisk.simulation import extForceTorque
    from Basilisk.simulation import simpleNav
    
    print(f"✓ Basilisk imported from: {bsk_path[0]}")
    print("="*70 + "\n")
except ImportError as e:
    print(f"✗ Basilisk import failed: {e}")
    print("\nMake sure Basilisk is installed:")
    print("  pip install avl-basilisk")
    sys.exit(1)

# ============================================================================
# CONFIGURABLE PARAMETERS
# ============================================================================

THRUST_NEWTONS = 1.0  # N - Now optimized for fast 1N simulation!
SPECIFIC_IMPULSE = 10000.0  # seconds
INITIAL_MASS = 1000.0  # kg
INITIAL_ALTITUDE = 500.0e3  # meters (500 km)
TARGET_ALTITUDE = 35000.0e3  # meters (35,000 km)
COAST_DURATION_DAYS = 14.0  # days

# Performance optimization
CONTROL_UPDATE_INTERVAL = 10.0  # seconds between thrust updates
DATA_STORAGE_INTERVAL = 60.0  # seconds between data points (reduces memory)

# ============================================================================
# MAIN SIMULATION FUNCTION
# ============================================================================

def run(thrust_N=None, show_plots=True):
    """
    Execute LEO to GEO low-thrust transfer
    """
    
    # Use provided thrust or default
    if thrust_N is None:
        thrust_N = THRUST_NEWTONS
    
    print("\n" + "="*70)
    print(f"RUNNING SIMULATION WITH THRUST = {thrust_N} N")
    print("="*70)
    
    # Earth parameters
    R_E = 6.378e6  # m
    mu = 3.986004418e14  # m^3/s^2
    
    # Orbit parameters
    a_init = INITIAL_ALTITUDE
    r_init = R_E + a_init
    v_init = np.sqrt(mu / r_init)
    
    a_final = TARGET_ALTITUDE
    r_final = R_E + a_final
    
    # Spacecraft parameters
    m_0 = INITIAL_MASS
    T_max = thrust_N
    I_sp = SPECIFIC_IMPULSE
    g_0 = 9.80665
    
    print(f"Initial Altitude:    {a_init/1e3:.1f} km")
    print(f"Target Altitude:     {a_final/1e3:.1f} km")
    print(f"Thrust:              {T_max:.1f} N")
    print(f"Specific Impulse:    {I_sp:.0f} s")
    print(f"Initial Mass:        {m_0:.1f} kg")
    print("="*70)
    
    # Create simulation
    print("\nInitializing Basilisk simulation...")
    scSim = SimulationBaseClass.SimBaseClass()
    
    # Create process and task
    dynProcess = scSim.CreateNewProcess("dynamicsProcess")
    simTimeStep = macros.sec2nano(1.0)
    dynProcess.addTask(scSim.CreateNewTask("dynamicsTask", simTimeStep))
    
    # Spacecraft
    print("Setting up spacecraft...")
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "spacecraft"
    scObject.hub.mHub = m_0
    scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
    
    I_tensor = [900., 0., 0., 0., 800., 0., 0., 0., 600.]
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I_tensor)
    
    scObject.hub.r_CN_NInit = [[r_init], [0.0], [0.0]]
    scObject.hub.v_CN_NInit = [[0.0], [v_init], [0.0]]
    scObject.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]
    scObject.hub.omega_BN_BInit = [[0.0], [0.0], [0.0]]
    
    scSim.AddModelToTask("dynamicsTask", scObject)
    
    # Gravity
    print("Setting up gravity...")
    gravFactory = simIncludeGravBody.gravBodyFactory()
    earth = gravFactory.createEarth()
    earth.isCentralBody = True
    earth.mu = mu
    gravFactory.addBodiesTo(scObject)
    
    # External force
    print("Setting up thrust system...")
    extForce = extForceTorque.ExtForceTorque()
    extForce.ModelTag = "thrustForce"
    scObject.addDynamicEffector(extForce)
    scSim.AddModelToTask("dynamicsTask", extForce)
    
    # Navigation
    navObject = simpleNav.SimpleNav()
    navObject.ModelTag = "navigation"
    navObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
    scSim.AddModelToTask("dynamicsTask", navObject)
    
    # Data logging
    dataLog = scObject.scStateOutMsg.recorder(simTimeStep)
    scSim.AddModelToTask("dynamicsTask", dataLog)
    
    # Initialize
    print("Initializing simulation...")
    scSim.InitializeSimulation()
    print("✓ Simulation initialized\n")
    
    # ========================================================================
    # THRUST PHASE
    # ========================================================================
    
    print(">>> Starting Thrust Phase...")
    
    MAX_SIM_DAYS = 1095 # 3 years safety limit for long runs --> can change later
    sim_time = 0.0
    max_sim_time = MAX_SIM_DAYS * 86400.0  # Use configurable max time
    control_update = 10.0
    
    thrust_active = True
    orbit_crossings = 0
    previous_y = 0.0
    
    thrust_r_history = []
    thrust_v_history = []
    thrust_t_history = []
    thrust_m_history = []
    
    step_count = 0
    print_interval = 1000  # Print more frequently for progress
    
    print(f"Max simulation time: {MAX_SIM_DAYS} days")
    print(f"Progress updates every {print_interval} steps (~{print_interval*control_update/3600:.1f} hours)\n")
    
    while sim_time < max_sim_time and thrust_active:
        
        scSim.ConfigureStopTime(macros.sec2nano(sim_time + control_update))
        scSim.ExecuteSimulation()
        
        r_BN = scObject.dynManager.getStateObject("hubPosition").getState()
        v_BN = scObject.dynManager.getStateObject("hubVelocity").getState()
        mass = scObject.hub.mHub
        
        r_vec = np.array([r_BN[0][0], r_BN[1][0], r_BN[2][0]])
        v_vec = np.array([v_BN[0][0], v_BN[1][0], v_BN[2][0]])
        
        r_mag = np.linalg.norm(r_vec)
        v_mag = np.linalg.norm(v_vec)
        
        thrust_r_history.append(r_vec.copy())
        thrust_v_history.append(v_vec.copy())
        thrust_t_history.append(sim_time)
        thrust_m_history.append(mass)
        
        if previous_y < 0 and r_vec[1] >= 0 and sim_time > 0:
            orbit_crossings += 1
        previous_y = r_vec[1]
        
        step_count += 1
        
        if r_mag >= r_final:
            thrust_active = False
            print(f"✓ Target reached at t={sim_time/86400.0:.4f} days")
            break
        
        if v_mag > 0:
            thrust_direction = v_vec / v_mag
            thrust_force = T_max * thrust_direction
            extForce.extForce_N = [[thrust_force[0]], [thrust_force[1]], [thrust_force[2]]]
        else:
            extForce.extForce_N = [[0.0], [0.0], [0.0]]
        
        delta_t = control_update
        delta_m = -T_max / (I_sp * g_0) * delta_t
        scObject.hub.mHub = max(10.0, mass + delta_m)
        
        sim_time += control_update
    
    # Check if simulation timed out
    if sim_time >= max_sim_time and thrust_active:
        print(f"⚠ Simulation stopped at max safety limit")
        print(f"  Final altitude: {(r_mag - R_E)/1e3:.1f} km")
        print(f"  Target altitude: {a_final/1e3:.1f} km")
        # Skip coast phase if we didn't reach target
        show_coast = False
    else:
        show_coast = True
    
    thrust_end_time = sim_time
    final_mass_thrust = scObject.hub.mHub
    fuel_used = m_0 - final_mass_thrust
    
    v_BN = scObject.dynManager.getStateObject("hubVelocity").getState()
    v_final_thrust = np.linalg.norm([v_BN[0][0], v_BN[1][0], v_BN[2][0]])
    
    print(f"\nThrust Phase Complete:")
    print(f"  Duration:        {thrust_end_time/86400.0:.4f} days")
    print(f"  Orbits:          {orbit_crossings}")
    print(f"  Fuel used:       {fuel_used:.4f} kg")
    print(f"  Final velocity:  {v_final_thrust:.3f} m/s")
    
    # ========================================================================
    # COAST PHASE
    # ========================================================================
    
    if show_coast:
        print(f"\n>>> Starting Coast Phase ({COAST_DURATION_DAYS} days)...")
        
        extForce.extForce_N = [[0.0], [0.0], [0.0]]
        
        coast_duration = COAST_DURATION_DAYS * 86400.0
        coast_end_time = thrust_end_time + coast_duration
        coast_sample_rate = 600.0
        
        coast_r_history = []
        coast_v_history = []
        coast_t_history = []
        
        while sim_time < coast_end_time:
            next_time = min(sim_time + coast_sample_rate, coast_end_time)
            
            scSim.ConfigureStopTime(macros.sec2nano(next_time))
            scSim.ExecuteSimulation()
            
            r_BN = scObject.dynManager.getStateObject("hubPosition").getState()
            v_BN = scObject.dynManager.getStateObject("hubVelocity").getState()
            
            r_vec = np.array([r_BN[0][0], r_BN[1][0], r_BN[2][0]])
            v_vec = np.array([v_BN[0][0], v_BN[1][0], v_BN[2][0]])
            
            coast_r_history.append(r_vec.copy())
            coast_v_history.append(v_vec.copy())
            coast_t_history.append(sim_time)
            
            sim_time = next_time
        
        print(f"✓ Coast phase complete")
        
        coast_radii = [np.linalg.norm(r) for r in coast_r_history]
        r_apogee = max(coast_radii)
        r_perigee = min(coast_radii)
        r_mean = (r_apogee + r_perigee) / 2.0
        eccentricity = (r_apogee - r_perigee) / (r_apogee + r_perigee)
        radius_error_pct = (r_mean - r_final) / r_final * 100.0
        
        v_final_coast = np.linalg.norm(coast_v_history[-1])
        
        print(f"  Radius error:    {radius_error_pct:.3f}%")
        print(f"  Eccentricity:    {eccentricity:.5f}")
        print(f"  Final velocity:  {v_final_coast:.3f} m/s")
    else:
        coast_r_history = []
        coast_t_history = []
        coast_v_history = []
        eccentricity = 0.0
        radius_error_pct = 0.0
        v_final_coast = v_final_thrust
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    print(f"Thrust:             {T_max:.1f} N")
    print(f"Transfer time:      {thrust_end_time/86400.0:.4f} days")
    print(f"Fuel used:          {fuel_used:.4f} kg")
    print(f"Orbits:             {orbit_crossings}")
    print(f"Coast error:        {radius_error_pct:.3f}%")
    print(f"Coast eccentricity: {eccentricity:.5f}")
    print("="*70 + "\n")
    
    # ========================================================================
    # PLOTTING
    # ========================================================================
    
    if show_plots:
        print("Generating plots...")
        
        thrust_r_history = np.array(thrust_r_history)
        coast_r_history = np.array(coast_r_history) if show_coast else np.array([])
        thrust_v_history = np.array(thrust_v_history)
        coast_v_history = np.array(coast_v_history) if show_coast else np.array([])
        thrust_t_history = np.array(thrust_t_history)
        coast_t_history = np.array(coast_t_history) if show_coast else np.array([])
        
        # Create detailed summary text with all parameters
        summary_text = (
            f"MISSION PARAMETERS\n"
            f"{'─'*40}\n"
            f"Initial Altitude:     {a_init/1e3:.1f} km\n"
            f"Target Altitude:      {a_final/1e3:.1f} km\n"
            f"Max Thrust:           {T_max:.1f} N\n"
            f"Specific Impulse:     {I_sp:.0f} s\n"
            f"Initial Mass:         {m_0:.1f} kg\n"
            f"\n"
            f"PERFORMANCE RESULTS\n"
            f"{'─'*40}\n"
            f"Initial Speed:        {v_init:.2f} m/s\n"
            f"Final Speed (thrust): {v_final_thrust:.2f} m/s\n"
            f"Final Speed (coast):  {v_final_coast:.2f} m/s\n"
            f"Fuel Used:            {fuel_used:.2f} kg ({fuel_used/m_0*100:.1f}%)\n"
            f"Transfer Time:        {thrust_end_time/86400.0:.2f} days\n"
            f"Number of Orbits:     {orbit_crossings}\n"
            f"\n"
            f"COAST PHASE ANALYSIS\n"
            f"{'─'*40}\n"
            f"Coast Duration:       {COAST_DURATION_DAYS:.0f} days\n"
            f"Radius Error:         {radius_error_pct:.3f}%\n"
            f"Eccentricity:         {eccentricity:.6f}"
        )
        
        # ====================================================================
        # PLOT 1: TRAJECTORY (X-Y Plane) with detailed parameters
        # ====================================================================
        plt.rc("font", size=12)
        fig1 = plt.figure(figsize=(14, 10))
        
        # Create grid: trajectory on left, text box on right
        gs = fig1.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.3)
        ax1 = fig1.add_subplot(gs[0])
        ax_text = fig1.add_subplot(gs[1])
        
        ax1.set_aspect("equal")
        ax1.axis("off")
        
        earth_patch = Circle((0, 0), R_E/1e3, ec="none", fc="C0")
        ax1.add_patch(earth_patch)
        ax1.annotate("Earth", xy=(0, 0), ha="center", va="center", color="white", fontsize=10)
        
        target_patch = Circle((0, 0), r_final/1e3, ec="C1", fc="none", lw=2, ls="--", label="Target Orbit")
        ax1.add_patch(target_patch)
        
        ax1.plot(thrust_r_history[:, 0]/1e3, thrust_r_history[:, 1]/1e3,
                color="C2", lw=1.5, label="Thrust Phase")
        if show_coast and len(coast_r_history) > 0:
            ax1.plot(coast_r_history[:, 0]/1e3, coast_r_history[:, 1]/1e3,
                    color="C3", lw=1.5, label="Coast Phase")
        
        ax1.legend(loc="upper right", fontsize=9)
        ax1.set_title(f"LEO to GEO Transfer Trajectory", fontsize=14, fontweight='bold', pad=15)
        
        # Text box with all parameters
        ax_text.axis('off')
        ax_text.text(0.05, 0.95, summary_text, transform=ax_text.transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='left',
                    fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.8", facecolor="lightgray", alpha=0.3, edgecolor='gray'))
        plt.tight_layout()
        
        # ====================================================================
        # PLOT 2: ALTITUDE VS TIME
        # ====================================================================
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        
        # Calculate altitudes
        thrust_alt_km = np.array([np.linalg.norm(r) - R_E for r in thrust_r_history]) / 1e3
        thrust_time_days = thrust_t_history / 86400.0
        
        # Plot thrust phase
        ax2.plot(thrust_time_days, thrust_alt_km, 'g-', linewidth=2, label='Thrust Phase')
        
        # Plot coast phase if available
        if show_coast and len(coast_r_history) > 0:
            coast_alt_km = np.array([np.linalg.norm(r) - R_E for r in coast_r_history]) / 1e3
            coast_time_days = coast_t_history / 86400.0
            ax2.plot(coast_time_days, coast_alt_km, 'r-', linewidth=2, label='Coast Phase')
        
        # Target altitude line
        ax2.axhline(y=a_final/1e3, color='b', linestyle='--', linewidth=1.5, label='Target Altitude', alpha=0.7)
        
        ax2.set_xlabel('Time (days)', fontsize=12)
        ax2.set_ylabel('Altitude (km)', fontsize=12)
        ax2.set_title(f'Altitude vs Time ({T_max} N Thrust)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        plt.tight_layout()
        
        # ====================================================================
        # PLOT 3: VELOCITY MAGNITUDE VS TIME
        # ====================================================================
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        
        # Calculate velocity magnitudes
        thrust_vel_mag = np.array([np.linalg.norm(v) for v in thrust_v_history])
        
        # Plot thrust phase
        ax3.plot(thrust_time_days, thrust_vel_mag, 'g-', linewidth=2, label='Thrust Phase')
        
        # Plot coast phase if available
        if show_coast and len(coast_v_history) > 0:
            coast_vel_mag = np.array([np.linalg.norm(v) for v in coast_v_history])
            ax3.plot(coast_time_days, coast_vel_mag, 'r-', linewidth=2, label='Coast Phase')
        
        ax3.set_xlabel('Time (days)', fontsize=12)
        ax3.set_ylabel('Velocity (m/s)', fontsize=12)
        ax3.set_title(f'Velocity Magnitude vs Time ({T_max} N Thrust)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10)
        plt.tight_layout()
        
        plt.show()
        
        print("✓ All plots displayed")
    
    return scSim


# ============================================================================
# RUN SIMULATION
# ============================================================================

if __name__ == "__main__":
    print("\nStarting simulation with default parameters...")
    run(show_plots=True)
    print("\n✓ Script complete!")
