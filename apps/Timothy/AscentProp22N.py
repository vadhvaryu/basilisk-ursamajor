import numpy as np
import matplotlib.pyplot as plt

# Basilisk Imports
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport
from Basilisk.utilities import simIncludeGravBody
from Basilisk.simulation import spacecraft
from Basilisk.simulation import extForceTorque

def run_chemical_simulation(propellant_mass=550.0, dry_mass=250.0, show_plots=True):
    # 1. Create Simulation Container
    scSim = SimulationBaseClass.SimBaseClass()
    dynProcess = scSim.CreateNewProcess("dynProcess")
    
    # Step Size: 10 seconds
    simulationTimeStep = macros.sec2nano(10.0) 
    dynProcess.addTask(scSim.CreateNewTask("dynTask", simulationTimeStep))

    # 2. Create Spacecraft Mass Properties
    m_0 = dry_mass + propellant_mass
    
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "Chem-Sat"
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
    req = 6378137.0 
    gravFactory.addBodiesTo(scObject)

    # 4. Thruster Setup
    extFTObject = extForceTorque.ExtForceTorque()
    extFTObject.ModelTag = "ASCENT-Macaw-22N"
    scObject.addDynamicEffector(extFTObject)
    scSim.AddModelToTask("dynTask", extFTObject)

    # --- ADVANCED ASCENT THRUSTER (Benchmark Macaw Specs) ---
    thrust_mag = 22.0   # 22 Newtons
    real_isp = 266.0    # ASCENT Isp at 300psi
    mdot = thrust_mag / (real_isp * 9.81)
    
    print(f"--- ASCENT Multi-Perigee Kick Simulation ---")
    print(f"Thruster: 22-Newton Macaw Class")
    print(f"Total Initial Mass: {m_0} kg (Dry: {dry_mass} kg, Fuel: {propellant_mass} kg)")
    print(f"Thrust: {thrust_mag} N | Isp: {real_isp} s | Mass Flow: {mdot:.5f} kg/s")

    # 5. Initial Orbit (Circular LEO at 400 km)
    mu = earth.mu
    a = req + 400000.0  
    e = 0.0             
    i = 10.0 * macros.D2R 
    omega = 0.0
    Omega = 0.0
    f = 0.0 
    
    r_init, v_init = orbitalMotion_elem2rv(mu, a, e, i, omega, Omega, f)
    
    scObject.hub.r_CN_NInit = [[r_init[0]], [r_init[1]], [r_init[2]]]
    scObject.hub.v_CN_NInit = [[v_init[0]], [v_init[1]], [v_init[2]]]
    scObject.hub.sigma_BNInit = [[0.2], [-0.4], [0.1]] 
    scObject.hub.omega_BN_BInit = [[0.0008], [0.001], [-0.0005]]

    # 6. Initialize Sim
    scSim.InitializeSimulation()
    
    # Run for 120 hours (5 days) due to the very low thrust requiring many passes
    sim_duration_sec = 120.0 * 3600.0
    
    current_time_nano = 0
    update_interval_sec = 10.0 
    update_interval_nano = macros.sec2nano(update_interval_sec)
    
    data = {
        "r": [], "time_min": [], "thrust_active": [], "fuel_mass": []
    }
    
    current_fuel = propellant_mass
    # Widened burn window to +/- 30 degrees to ensure fuel is spent in a reasonable timeframe
    burn_window_deg = 30.0 
    
    print(f"Simulating {(sim_duration_sec/3600.0):.1f} hours of flight...")
    
    while current_time_nano < sim_duration_sec * 1e9:
        scSim.ConfigureStopTime(current_time_nano + update_interval_nano)
        scSim.ExecuteSimulation()
        
        r_BN = scObject.dynManager.getStateObject("hubPosition").getState()
        v_BN = scObject.dynManager.getStateObject("hubVelocity").getState()
        r_vec = np.array(r_BN).flatten()
        v_vec = np.array(v_BN).flatten()
        
        # Sub-sample data for plotting to save memory over 5 simulated days
        if (current_time_nano / 1e9) % 60 == 0:
            data["r"].append(r_vec)
            data["time_min"].append(current_time_nano * 1e-9 / 60.0)
            data["fuel_mass"].append(current_fuel)
        
        # --- PERIGEE DETECTION ---
        r_norm = np.linalg.norm(r_vec)
        v_norm = np.linalg.norm(v_vec)
        
        e_vec = ((v_norm**2 - mu/r_norm)*r_vec - np.dot(r_vec, v_vec)*v_vec) / mu
        ecc = np.linalg.norm(e_vec)
        
        if ecc < 1e-5:
            true_anomaly = 0.0 # Assume perigee if practically circular
        else:
            cos_f = np.clip(np.dot(e_vec, r_vec) / (ecc * r_norm), -1.0, 1.0)
            true_anomaly = np.arccos(cos_f)
            if np.dot(r_vec, v_vec) < 0:
                true_anomaly = 2 * np.pi - true_anomaly
                
        f_deg = np.degrees(true_anomaly)
        
        # --- MULTI-PASS BURN LOGIC ---
        in_burn_window = (f_deg < burn_window_deg) or (f_deg > 360 - burn_window_deg)
        
        if current_fuel > 0 and in_burn_window:
            thrust_dir = v_vec / v_norm # Prograde thrust direction
            f_apply = thrust_dir * thrust_mag
            
            fuel_burned = mdot * update_interval_sec
            if current_fuel < fuel_burned:
                f_apply = f_apply * (current_fuel / fuel_burned)
                current_fuel = 0.0
            else:
                current_fuel -= fuel_burned
                
            # Update mass only if we took a data snapshot this frame to avoid index errors
            if (current_time_nano / 1e9) % 60 == 0:
                scObject.hub.mHub -= (data["fuel_mass"][-1] - current_fuel) 
                data["thrust_active"].append(1)
        else:
            f_apply = np.array([0.0, 0.0, 0.0]) # Coasting phase
            if (current_time_nano / 1e9) % 60 == 0:
                data["thrust_active"].append(0)
            
        extFTObject.extForce_N = [[f_apply[0]], [f_apply[1]], [f_apply[2]]]
        current_time_nano += update_interval_nano

    # --- PLOTTING ---
    if show_plots:
        print("Generating Plots...")
        r_data = np.array(data["r"])
        thrust_active = np.array(data["thrust_active"])
        
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
        
        burn_mask = thrust_active == 1
        coast_mask = thrust_active == 0
        
        ax1.plot(r_data[coast_mask, 0]/1000, r_data[coast_mask, 1]/1000, 'b.', markersize=0.5, label='Coasting (Transfer)', alpha=0.3)
        ax1.plot(r_data[burn_mask, 0]/1000, r_data[burn_mask, 1]/1000, 'r.', markersize=1.5, label='22N Engine Firing')
        
        ax1.plot(0, 0, 'bo', markersize=10, label="Earth") 
        earth_c = plt.Circle((0,0), req/1000, color='b', alpha=0.1)
        ax1.add_patch(earth_c)
        
        GEO_radius_km = 42164.0
        geo_c = plt.Circle((0,0), GEO_radius_km, color='g', fill=False, linestyle='--', alpha=0.5, label='GEO Altitude')
        ax1.add_patch(geo_c)

        ax1.set_title(f'Figure 1: LEO to GTO via 22-N ASCENT Macaw Thruster')
        ax1.set_xlabel('$R_x$ (km)')
        ax1.set_ylabel('$R_y$ (km)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        plt.tight_layout()
        plt.savefig('Macaw_22N_Trajectory.png', dpi=300)
        print("Saved: Macaw_22N_Trajectory.png")
        plt.show()

# --- HELPER FUNCTION ---
def orbitalMotion_elem2rv(mu, a, e, i, omega, Omega, f):
    if e < 1e-10:
        p = a
    else:
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
    # Test with standard variables: 550kg fuel, 250kg dry mass
    run_chemical_simulation(propellant_mass=550.0, dry_mass=250.0)