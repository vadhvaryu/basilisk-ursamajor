
import numpy as np
import matplotlib.pyplot as plt

# Basilisk Imports
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport
from Basilisk.utilities import simIncludeGravBody
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.simulation import spacecraft
from Basilisk.simulation import extForceTorque

def run_chemical_simulation(show_plots=True):
    # 1. Create Simulation Container
    scSim = SimulationBaseClass.SimBaseClass()
    dynProcess = scSim.CreateNewProcess("dynProcess")
    
    # Step Size: Back to 10 seconds for high-accuracy during the fast burn
    simulationTimeStep = macros.sec2nano(10.0) 
    dynProcess.addTask(scSim.CreateNewTask("dynTask", simulationTimeStep))

    # 2. Create Spacecraft 
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "Chem-Sat"
    m_0 = 3500.0 
    scObject.hub.mHub = m_0
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
    extFTObject.ModelTag = "Biprop-400N"
    scObject.addDynamicEffector(extFTObject)
    scSim.AddModelToTask("dynTask", extFTObject)

    # --- CHEMICAL THRUSTER SETUP ---
    # Standard Liquid Apogee Engine (Bipropellant)
    thrust_mag = 400.0  # 400 Newtons of force
    real_isp = 320.0    # Lower efficiency than electric
    mdot = thrust_mag / (real_isp * 9.81)
    
    # Burn duration: 15 minutes
    burn_duration_sec = 15.0 * 60.0
    burn_duration_nano = macros.sec2nano(burn_duration_sec)

    print(f"Thruster: Chemical Bipropellant Engine")
    print(f"Thrust: {thrust_mag} N")
    print(f"Burn Duration: {burn_duration_sec / 60.0} minutes")

    # 5. Initial Orbit 
    mu = earth.mu
    req = 6378137.0 
    
    a = req + 2000000.0 
    e = 0.3              
    i = 10.0 * macros.D2R 
    omega = 0.0
    Omega = 0.0
    f = 0.0 # Starting exactly at perigee (closest point to Earth)
    
    r_init, v_init = orbitalMotion_elem2rv(mu, a, e, i, omega, Omega, f)
    
    scObject.hub.r_CN_NInit = [[r_init[0]], [r_init[1]], [r_init[2]]]
    scObject.hub.v_CN_NInit = [[v_init[0]], [v_init[1]], [v_init[2]]]

    # Attitude Initialization
    scObject.hub.sigma_BNInit = [[0.2], [-0.4], [0.1]] 
    scObject.hub.omega_BN_BInit = [[0.0008], [0.001], [-0.0005]]

    # 6. Initialize Sim
    scSim.InitializeSimulation()
    
    # Run for just 6 hours to see the new trajectory post-burn
    sim_duration_sec = 6.0 * 3600.0
    
    current_time_nano = 0
    update_interval_sec = 10.0 # High resolution plotting
    update_interval_nano = macros.sec2nano(update_interval_sec)
    
    data = {
        "r": [],
        "time_min": [],
        "proj1": [], 
        "proj2": [], 
        "proj3": [],
        "thrust_active": []
    }
    
    print(f"Simulating {(sim_duration_sec/3600.0):.1f} hours of flight...")
    
    while current_time_nano < sim_duration_sec * 1e9:
        scSim.ConfigureStopTime(current_time_nano + update_interval_nano)
        scSim.ExecuteSimulation()
        
        # --- EXTRACT DATA ---
        r_BN = scObject.dynManager.getStateObject("hubPosition").getState()
        v_BN = scObject.dynManager.getStateObject("hubVelocity").getState()
        sigma_BN = scObject.dynManager.getStateObject("hubSigma").getState()
        
        r_vec = np.array(r_BN).flatten()
        v_vec = np.array(v_BN).flatten()
        sigma = np.array(sigma_BN).flatten()
        
        # --- ORIENTATION MATH ---
        r_norm = np.linalg.norm(r_vec)
        i_r = r_vec / r_norm
        h_vec = np.cross(r_vec, v_vec)
        i_h = h_vec / np.linalg.norm(h_vec)
        i_theta = np.cross(i_h, i_r)
        
        dcm_BN = rbk.MRP2C(sigma) 
        b1 = dcm_BN[0, :]
        b2 = dcm_BN[1, :]
        b3 = dcm_BN[2, :]
        
        data["proj1"].append(np.dot(b1, i_r))      
        data["proj2"].append(np.dot(b2, i_theta)) 
        data["proj3"].append(np.dot(b3, i_h))     
        
        # --- STORE DATA ---
        data["r"].append(r_vec)
        data["time_min"].append(current_time_nano * 1e-9 / 60.0)
        
        # --- APPLY IMPULSIVE BURN LOGIC ---
        # Only thrust if we are within the 15-minute burn window
        if current_time_nano < burn_duration_nano:
            v_norm = np.linalg.norm(v_vec)
            thrust_dir = v_vec / v_norm # Prograde burn
            f_apply = thrust_dir * thrust_mag
            scObject.hub.mHub -= (mdot * update_interval_sec) # Burn fuel
            data["thrust_active"].append(1)
        else:
            f_apply = np.array([0.0, 0.0, 0.0]) # Engine Cutoff (MECO)
            data["thrust_active"].append(0)
            
        extFTObject.extForce_N = [[f_apply[0]], [f_apply[1]], [f_apply[2]]]
        
        current_time_nano += update_interval_nano

    # --- PLOTTING ---
    if show_plots:
        print("Generating Plots...")
        r_data = np.array(data["r"])
        t_data = np.array(data["time_min"])
        thrust_active = np.array(data["thrust_active"])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 12))
        
        # Top Plot: Orbit (Highlighting the burn phase)
        burn_mask = thrust_active == 1
        coast_mask = thrust_active == 0
        
        # Plot coasting phase
        ax1.plot(r_data[coast_mask, 0]/1000, r_data[coast_mask, 1]/1000, 'b-', linewidth=1.5, label='Coasting')
        # Plot burn phase in red
        ax1.plot(r_data[burn_mask, 0]/1000, r_data[burn_mask, 1]/1000, 'r-', linewidth=3.0, label='15-Min Engine Burn')
        
        ax1.plot(0, 0, 'bo', markersize=10, label="Earth") 
        earth_c = plt.Circle((0,0), req/1000, color='b', alpha=0.1)
        ax1.add_patch(earth_c)
        
        ax1.set_title(f'Figure 1: Chemical Perigee Kick (Prograde)')
        ax1.set_xlabel('$R_x$ (km)')
        ax1.set_ylabel('$R_y$ (km)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Bottom Plot: Orientation
        ax2.plot(t_data, data["proj1"], label=r'$\hat{i}_r \cdot \hat{b}_1$ (Radial)', color='C0', alpha=0.7)
        ax2.plot(t_data, data["proj2"], label=r'$\hat{i}_{\theta} \cdot \hat{b}_2$ (Transverse)', color='C1', alpha=0.7)
        ax2.plot(t_data, data["proj3"], label=r'$\hat{i}_h \cdot \hat{b}_3$ (Normal)', color='C2', alpha=0.7)
        
        # Add a shaded region to show when the engine is firing
        ax2.axvspan(0, 15, color='red', alpha=0.1, label='Engine Firing')
        
        ax2.set_title('Figure 2: Orientation Illustration (Tumbling Spacecraft)')
        ax2.set_xlabel('Time [min]')
        ax2.set_ylabel('Orientation Projection')
        ax2.set_ylim(-1.2, 1.2)
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('Chemical_Orbit_Analysis.png', dpi=300)
        print("Saved: Chemical_Orbit_Analysis.png")
        plt.show()

# --- MATH HELPER FOR ELLIPTICAL ORBITS ---
def orbitalMotion_elem2rv(mu, a, e, i, omega, Omega, f):
    p = a * (1 - e*e)
    r = p / (1 + e * np.cos(f))
    
    r_PQW = np.array([r * np.cos(f), r * np.sin(f), 0.0])
    v_PQW = np.array([-np.sqrt(mu/p)*np.sin(f), np.sqrt(mu/p)*(e + np.cos(f)), 0.0])
    
    R3_W = np.array([[np.cos(Omega), np.sin(Omega), 0], [-np.sin(Omega), np.cos(Omega), 0], [0, 0, 1]])
    R1_i = np.array([[1, 0, 0], [0, np.cos(i), np.sin(i)], [0, -np.sin(i), np.cos(i)]])
    R3_w = np.array([[np.cos(omega), np.sin(omega), 0], [-np.sin(omega), np.cos(omega), 0], [0, 0, 1]])
    
    Q_Px = R3_w @ R1_i @ R3_W
    Q_xP = np.transpose(Q_Px)
    
    return Q_xP @ r_PQW, Q_xP @ v_PQW

if __name__ == "__main__":
    run_chemical_simulation()hvaryu/basilisk-ursamajor/tree/main