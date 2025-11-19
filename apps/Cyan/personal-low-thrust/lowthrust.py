"""
Scenario: Nonimpulsive low-thrust LEO -> GEO transfer using Basilisk 2.8.19
- Single thruster firing opposite to velocity direction (forward thrust)
- Stop thrust when reaching GEO radius
- Coast for 2 weeks to verify orbit stability
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Add Basilisk to path (adjust as needed for your installation)
# repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
# basilisk_path = os.path.join(repo_root, "basilisk")
# if basilisk_path not in sys.path:
#     sys.path.append(basilisk_path)

# Basilisk imports
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.simulation import spacecraft
from Basilisk.simulation import thrusterDynamicEffector
from Basilisk.utilities import simIncludeGravBody
from Basilisk import __path__

bskPath = __path__[0]

# ============================================================
# INITIAL PARAMETERS
# ============================================================

# Gravitational parameter and Earth radius
MU_EARTH = 3.986004418e14  # m^3/s^2
R_EARTH = 6.378e6  # m

# Initial orbit (LEO)
ALT_LEO = 500e3  # m (500 km altitude)
R_LEO = R_EARTH + ALT_LEO
V_LEO = np.sqrt(MU_EARTH / R_LEO)  # Circular orbit velocity

# Target orbit (GEO)
ALT_GEO = 35_786e3  # m (35,786 km altitude)
R_GEO = R_EARTH + ALT_GEO

# Spacecraft parameters
MASS_INIT = 1000.0  # kg
MAX_THRUST = 1.0  # N
ISP = 10_000.0  # seconds
G0 = 9.80665  # m/s^2

# Simulation parameters
TIME_STEP = 10.0  # seconds
COAST_DURATION = 14 * 24 * 3600.0  # 2 weeks in seconds
MAX_SIM_TIME = 3.0 * 365 * 24 * 3600.0  # 3 years safety limit

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def unit_vector(vec):
    """Return unit vector in direction of vec"""
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        return vec / norm
    return np.array([1.0, 0.0, 0.0])

# ============================================================
# MAIN SCENARIO FUNCTION
# ============================================================

def run_leo_to_geo_transfer(show_plots=True):
    """
    Simulate low-thrust transfer from LEO to GEO with coast phase
    """
    
    # --------------------------------------------------------
    # Create simulation container
    scSim = SimulationBaseClass.SimBaseClass()
    
    # --------------------------------------------------------
    # Create simulation process and task
    dynProcess = scSim.CreateNewProcess("dynamicsProcess")
    simulationTimeStep = macros.sec2nano(TIME_STEP)
    dynProcess.addTask(scSim.CreateNewTask("dynamicsTask", simulationTimeStep))
    
    # --------------------------------------------------------
    # Create spacecraft object
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "spacecraft"
    
    # Set initial mass
    scObject.hub.mHub = MASS_INIT
    
    # Set initial position (LEO, starting on x-axis)
    scObject.hub.r_CN_NInit = [[R_LEO], [0.0], [0.0]]
    
    # Set initial velocity (circular orbit in y-direction)
    scObject.hub.v_CN_NInit = [[0.0], [V_LEO], [0.0]]
    
    # Set initial attitude (identity - not critical for this simulation)
    scObject.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]
    
    # Add spacecraft to simulation
    scSim.AddModelToTask("dynamicsTask", scObject)
    
    # --------------------------------------------------------
    # Setup gravity body (Earth)
    gravFactory = simIncludeGravBody.gravBodyFactory()
    earth = gravFactory.createEarth()
    earth.isCentralBody = True
    earth.useSphericalHarmParams = False  # Simple point mass gravity
    
    # Attach gravity to spacecraft
    scObject.gravField.gravBodies = spacecraft.GravBodyVector(
        list(gravFactory.gravBodies.values())
    )
    
    # --------------------------------------------------------
    # Setup thruster
    thrusterSet = thrusterDynamicEffector.ThrusterDynamicEffector()
    thrusterSet.ModelTag = "thrusterDynamics"
    
    # Create single thruster manually
    thConfig = thrusterDynamicEffector.THRSimConfig()
    thConfig.thrLoc_B = [[0.0], [0.0], [0.0]]  # Location at COM
    thConfig.thrDir_B = [[1.0], [0.0], [0.0]]  # Direction (will be updated each timestep)
    thConfig.MaxThrust = MAX_THRUST  # Newtons
    thConfig.steadyIsp = ISP  # seconds
    
    # Add thruster to effector
    thrusterSet.addThruster(thConfig)
    
    # Link thruster to spacecraft mass depletion
    thrusterSet.linkInStates(scObject.scStateOutMsg)
    scObject.addDynamicEffector(thrusterSet)
    
    # Add thruster to simulation
    scSim.AddModelToTask("dynamicsTask", thrusterSet)
    
    # --------------------------------------------------------
    # Setup data logging
    numDataPoints = 1000000
    samplingTime = simulationTimeStep
    scLog = scObject.scStateOutMsg.recorder(samplingTime)
    scSim.AddModelToTask("dynamicsTask", scLog)
    
    # --------------------------------------------------------
    # Initialize simulation
    scSim.InitializeSimulation()
    
    # --------------------------------------------------------
    # MAIN SIMULATION LOOP
    print("Starting simulation...")
    print(f"Initial altitude: {ALT_LEO/1000:.1f} km")
    print(f"Target altitude: {ALT_GEO/1000:.1f} km")
    print(f"Max thrust: {MAX_THRUST} N")
    print(f"Isp: {ISP} s")
    
    # State tracking
    thrusting = True
    thrust_end_time = None
    current_time = 0.0
    step_count = 0
    max_steps = int(MAX_SIM_TIME / TIME_STEP)
    
    # Data storage
    time_history = []
    radius_history = []
    velocity_history = []
    mass_history = []
    
    while step_count < max_steps:
        # Get current state
        r_BN_N = scObject.dynManager.getStateObject("hubPosition").getState()
        v_BN_N = scObject.dynManager.getStateObject("hubVelocity").getState()
        current_mass = scObject.hub.mHub
        
        r_BN_N = np.array(r_BN_N).flatten()
        v_BN_N = np.array(v_BN_N).flatten()
        
        radius = np.linalg.norm(r_BN_N)
        
        # Store data
        time_history.append(current_time)
        radius_history.append(radius)
        velocity_history.append(np.linalg.norm(v_BN_N))
        mass_history.append(current_mass)
        
        # Check if we've reached GEO radius
        if thrusting and radius >= R_GEO:
            thrusting = False
            thrust_end_time = current_time
            print(f"\nThrust cutoff at t = {current_time/(24*3600):.2f} days")
            print(f"  Radius: {radius/1000:.1f} km")
            print(f"  Altitude: {(radius - R_EARTH)/1000:.1f} km")
            print(f"  Mass remaining: {current_mass:.2f} kg")
            print(f"  Propellant used: {MASS_INIT - current_mass:.2f} kg")
        
        # Set thrust command
        if thrusting:
            # Thrust in velocity direction (opposite to exhaust)
            v_hat = unit_vector(v_BN_N)
            
            # Update thruster direction
            thrusterSet.thrusterData[0].thrusterDirection = v_hat
            
            # Set thrust level to maximum
            thrusterSet.thrusterData[0].ThrustFactor = 1.0
        else:
            # Turn off thrust during coast
            thrusterSet.thrusterData[0].ThrustFactor = 0.0
        
        # Check if coast phase is complete
        if (not thrusting) and (thrust_end_time is not None):
            coast_elapsed = current_time - thrust_end_time
            if coast_elapsed >= COAST_DURATION:
                print(f"\nCoast phase complete at t = {current_time/(24*3600):.2f} days")
                print(f"  Coast duration: {coast_elapsed/(24*3600):.2f} days")
                break
        
        # Execute one simulation step
        scSim.ConfigureStopTime(macros.sec2nano(current_time + TIME_STEP))
        scSim.ExecuteSimulation()
        
        current_time += TIME_STEP
        step_count += 1
        
        # Progress reporting
        if step_count % 10000 == 0:
            print(f"  Step {step_count}: t = {current_time/(24*3600):.2f} days, "
                  f"r = {radius/1000:.1f} km, m = {current_mass:.2f} kg")
    
    # --------------------------------------------------------
    # POST-PROCESSING AND ANALYSIS
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    
    # Convert to numpy arrays
    time_history = np.array(time_history)
    radius_history = np.array(radius_history)
    velocity_history = np.array(velocity_history)
    mass_history = np.array(mass_history)
    
    # Extract thrust and coast phases
    if thrust_end_time is not None:
        thrust_mask = time_history <= thrust_end_time
        coast_mask = time_history > thrust_end_time
        
        # Analyze coast phase orbit
        coast_radii = radius_history[coast_mask]
        if len(coast_radii) > 0:
            r_apo = np.max(coast_radii)
            r_peri = np.min(coast_radii)
            r_mean = (r_apo + r_peri) / 2
            eccentricity = (r_apo - r_peri) / (r_apo + r_peri)
            error_percent = (r_mean - R_GEO) / R_GEO * 100
            
            print(f"\nCoast Phase Orbit Analysis:")
            print(f"  Apogee radius: {r_apo/1000:.1f} km")
            print(f"  Perigee radius: {r_peri/1000:.1f} km")
            print(f"  Mean radius: {r_mean/1000:.1f} km")
            print(f"  Target radius: {R_GEO/1000:.1f} km")
            print(f"  Radius error: {error_percent:.3f}%")
            print(f"  Eccentricity: {eccentricity:.6f}")
    
    print(f"\nFinal State:")
    print(f"  Time of flight: {time_history[-1]/(24*3600):.2f} days")
    print(f"  Final mass: {mass_history[-1]:.2f} kg")
    print(f"  Total propellant used: {MASS_INIT - mass_history[-1]:.2f} kg")
    
    # --------------------------------------------------------
    # PLOTTING
    if show_plots:
        # Get full trajectory from logged data
        r_logged = scLog.r_BN_N
        
        # Create trajectory plot
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')
        
        # Plot Earth
        earth_circle = Circle((0, 0), R_EARTH/1000, color='blue', 
                             label='Earth', zorder=10)
        ax.add_patch(earth_circle)
        
        # Plot GEO radius circle
        theta = np.linspace(0, 2*np.pi, 400)
        ax.plot(R_GEO/1000 * np.cos(theta), R_GEO/1000 * np.sin(theta), 
               'r--', linewidth=2, label='GEO Radius', alpha=0.7)
        
        # Plot trajectory
        if thrust_end_time is not None:
            # Separate thrust and coast phases
            thrust_points = r_logged[thrust_mask]
            coast_points = r_logged[coast_mask]
            
            ax.plot(thrust_points[:, 0]/1000, thrust_points[:, 1]/1000, 
                   'g-', linewidth=1.5, label='Thrust Phase', alpha=0.8)
            ax.plot(coast_points[:, 0]/1000, coast_points[:, 1]/1000, 
                   'orange', linewidth=1.5, label='Coast Phase', alpha=0.8)
        else:
            ax.plot(r_logged[:, 0]/1000, r_logged[:, 1]/1000, 
                   'g-', linewidth=1.5, label='Trajectory', alpha=0.8)
        
        ax.set_xlabel('X Position (km)', fontsize=12)
        ax.set_ylabel('Y Position (km)', fontsize=12)
        ax.set_title('Low-Thrust LEO to GEO Transfer', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return time_history, radius_history, velocity_history, mass_history

# ============================================================
# RUN SIMULATION

if __name__ == "__main__":
    run_leo_to_geo_transfer(show_plots=True)
