print("!!! XIPS-25 SIMULATION INITIALIZING !!!")

import numpy as np
import matplotlib.pyplot as plt

# Basilisk Imports
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport
from Basilisk.utilities import simIncludeGravBody
from Basilisk.utilities import RigidBodyKinematics as rbk # Essential for orientation math
from Basilisk.simulation import spacecraft
from Basilisk.simulation import extForceTorque

def run_xips_simulation(show_plots=True):
    # 1. Create Simulation Container
    scSim = SimulationBaseClass.SimBaseClass()
    dynProcess = scSim.CreateNewProcess("dynProcess")
    
    # Step Size: 10 seconds (Good balance of speed/accuracy)
    simulationTimeStep = macros.sec2nano(10.0) 
    dynProcess.addTask(scSim.CreateNewTask("dynTask", simulationTimeStep))

    # 2. Create Spacecraft (Boeing 702 Platform style)
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "Boeing-702"
    m_0 = 3500.0 # Heavier comms satellite
    scObject.hub.mHub = m_0
    # Large Inertia Tensor for a big satellite
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d([
        2000., 0., 0., 
        0., 1500., 0., 
        0., 0., 1200.
    ])
    scSim.AddModelToTask("dynTask", scObject)

    # 3. Gravity Setup
    gravFactory = simIncludeGravBody.gravBodyFactory()
    earth = gravFactory.createEarth()
    earth.isCentralBody = True
    earth.mu = 3.986004418e14
    gravFactory.addBodiesTo(scObject)

    # 4. Thruster Setup (External Force)
    extFTObject = extForceTorque.ExtForceTorque()
    extFTObject.ModelTag = "XIPS-25"
    scObject.addDynamicEffector(extFTObject)
    scSim.AddModelToTask("dynTask", extFTObject)

    # --- NEW THRUSTER: Boeing XIPS-25 ---
    # Reference: High power ion thruster
    # Real Force: 0.165 N (165 mN)
    # Real Isp: 3500 s
    real_thrust = 0.165
    real_isp = 3500.0
    
    # VISUALIZATION HACK: 
    # We multiply force by 100x so the graph generates in seconds, not hours.
    # The spiral shape remains accurate, just fewer loops.
    thrust_mag = real_thrust * 150.0 
    mdot = thrust_mag / (real_isp * 9.81)
    
    print(f"Thruster: XIPS-25 (Scaled for Visualization)")
    print(f"Simulated Force: {thrust_mag:.2f} N")

    # 5. Initial Orbit (Elliptical GTO-like start)
    # We use Classical Orbital Elements (COE) to define the shape
    mu = earth.mu
    req = 6378137.0 
    
    a = req + 2000000.0 # Semi-major axis
    e = 0.3             # Eccentricity (0.3 makes it an oval/ellipse)
    i = 10.0 * macros.D2R # 10 degree inclination
    omega = 0.0
    Omega = 0.0
    f = 0.0
    
    # Convert COE to Position/Velocity
    r_init, v_init = orbitalMotion_elem2rv(mu, a, e, i, omega, Omega, f)
    
    scObject.hub.r_CN_NInit = [[r_init[0]], [r_init[1]], [r_init[2]]]
    scObject.hub.v_CN_NInit = [[v_init[0]], [v_init[1]], [v_init[2]]]

    # --- ATTITUDE INITIALIZATION ---
    # To get the "Orientation Illustration" waves, the sat needs to be tumbling.
    # Initial Attitude (Random tumble)
    scObject.hub.sigma_BNInit = [[0.2], [-0.4], [0.1]] 
    # Initial Spin Rate (rad/s)
    scObject.hub.omega_BN_BInit = [[0.0008], [0.001], [-0.0005]]

    # 6. Initialize Sim
    scSim.InitializeSimulation()
    
    # Run for exactly 3 orbits to match your reference image style
    period = 2 * np.pi * np.sqrt(a**3 / mu)
    sim_duration = period * 3.5
    
    current_time_nano = 0
    # Sampling rate for plotting (every 60s)
    update_interval_sec = 60.0 
    update_interval_nano = macros.sec2nano(update_interval_sec)
    
    # Data Storage
    data = {
        "r": [],
        "time_min": [],
        "proj1": [], # Radial dot b1
        "proj2": [], # Theta dot b2
        "proj3": []  # Normal dot b3
    }
    
    print(f"Simulating {(sim_duration/3600.0):.1f} hours of flight...")
    
    while current_time_nano < sim_duration*1e9:
        scSim.ConfigureStopTime(current_time_nano + update_interval_nano)
        scSim.ExecuteSimulation()
        
        # --- EXTRACT DATA ---
        # Get Position, Velocity, Attitude
        r_BN = scObject.dynManager.getStateObject("hubPosition").getState()
        v_BN = scObject.dynManager.getStateObject("hubVelocity").getState()
        sigma_BN = scObject.dynManager.getStateObject("hubSigma").getState()
        
        r_vec = np.array(r_BN).flatten()
        v_vec = np.array(v_BN).flatten()
        sigma = np.array(sigma_BN).flatten()
        
        # --- ORIENTATION MATH (The "Hill Frame") ---
        # 1. Radial Vector (Unit vector pointing from Earth to Sat)
        r_norm = np.linalg.norm(r_vec)
        i_r = r_vec / r_norm
        
        # 2. Normal Vector (Cross product of r and v)
        h_vec = np.cross(r_vec, v_vec)
        i_h = h_vec / np.linalg.norm(h_vec)
        
        # 3. Transverse/Theta Vector (Cross product of h and r)
        i_theta = np.cross(i_h, i_r)
        
        # 4. Body Frame Vectors
        # Convert MRP attitude to Rotation Matrix [BN]
        dcm_BN = rbk.MRP2C(sigma) 
        b1 = dcm_BN[0, :] # Body X axis
        b2 = dcm_BN[1, :] # Body Y axis
        b3 = dcm_BN[2, :] # Body Z axis
        
        # 5. Dot Products (Project Body axes onto Orbit axes)
        # This generates the cosine waves in your reference image
        data["proj1"].append(np.dot(b1, i_r))     # How aligned is Body X with Radial?
        data["proj2"].append(np.dot(b2, i_theta)) # How aligned is Body Y with Velocity direction?
        data["proj3"].append(np.dot(b3, i_h))     # How aligned is Body Z with Orbit Normal?
        
        # --- STORE DATA ---
        data["r"].append(r_vec)
        data["time_min"].append(current_time_nano * 1e-9 / 60.0)
        
        # --- APPLY GUIDANCE (Thrust along Velocity) ---
        v_norm = np.linalg.norm(v_vec)
        thrust_dir = v_vec / v_norm
        f_apply = thrust_dir * thrust_mag
        extFTObject.extForce_N = [[f_apply[0]], [f_apply[1]], [f_apply[2]]]
        
        # Burn Fuel
        scObject.hub.mHub -= (mdot * update_interval_sec)
        
        current_time_nano += update_interval_nano

    # --- PLOTTING ---
    if show_plots:
        print("Generating Plots...")
        r_data = np.array(data["r"])
        t_data = np.array(data["time_min"])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 12))
        
        # Top Plot: Orbit
        ax1.plot(r_data[:, 0]/1000, r_data[:, 1]/1000, 'b-', linewidth=1.5)
        ax1.plot(0, 0, 'bo', markersize=10, label="Earth") # Earth Center
        # Draw Earth Circle
        earth_c = plt.Circle((0,0), req/1000, color='b', alpha=0.1)
        ax1.add_patch(earth_c)
        
        ax1.set_title(f'Figure 1: XIPS-25 Spiral (Eccentricity = {e})')
        ax1.set_xlabel('$R_x$ (km)')
        ax1.set_ylabel('$R_y$ (km)')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Bottom Plot: Orientation
        ax2.plot(t_data, data["proj1"], label=r'$\hat{i}_r \cdot \hat{b}_1$ (Radial)', color='C0')
        ax2.plot(t_data, data["proj2"], label=r'$\hat{i}_{\theta} \cdot \hat{b}_2$ (Transverse)', color='C1')
        ax2.plot(t_data, data["proj3"], label=r'$\hat{i}_h \cdot \hat{b}_3$ (Normal)', color='C2')
        
        ax2.set_title('Figure 2: Orientation Illustration (Tumbling Spacecraft)')
        ax2.set_xlabel('Time [min]')
        ax2.set_ylabel('Orientation Projection')
        ax2.set_ylim(-1.2, 1.2)
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('XIPS_Orbit_Analysis.png', dpi=300)
        print("Saved: XIPS_Orbit_Analysis.png")
        plt.show()

# --- MATH HELPER FOR ELLIPTICAL ORBITS ---
def orbitalMotion_elem2rv(mu, a, e, i, omega, Omega, f):
    # Standard orbital mechanics formulas to convert elements to vectors
    p = a * (1 - e*e)
    r = p / (1 + e * np.cos(f))
    
    r_PQW = np.array([r * np.cos(f), r * np.sin(f), 0.0])
    v_PQW = np.array([-np.sqrt(mu/p)*np.sin(f), np.sqrt(mu/p)*(e + np.cos(f)), 0.0])
    
    # Rotation Matrices
    R3_W = np.array([[np.cos(Omega), np.sin(Omega), 0], [-np.sin(Omega), np.cos(Omega), 0], [0, 0, 1]])
    R1_i = np.array([[1, 0, 0], [0, np.cos(i), np.sin(i)], [0, -np.sin(i), np.cos(i)]])
    R3_w = np.array([[np.cos(omega), np.sin(omega), 0], [-np.sin(omega), np.cos(omega), 0], [0, 0, 1]])
    
    Q_Px = R3_w @ R1_i @ R3_W
    Q_xP = np.transpose(Q_Px)
    
    return Q_xP @ r_PQW, Q_xP @ v_PQW

if __name__ == "__main__":
    run_xips_simulation()