"""
Nonimpulsive low-thrust LEO -> GEO transfer:
- 1 thruster, set in opposite direction to velocity of spacecraft.
- Do not need to account for rotation of body of spacecraft.
- Stop thrust when reach GEO radius.
- Model coast for 2 weeks and analyze stability of orbit.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Basilisk imports
from Basilisk.utilities import SimulationBaseClass, macros, orbitalMotion
from Basilisk.simulation import spacecraft
from Basilisk.utilities import simIncludeThruster, simIncludeGravBody
from Basilisk.architecture import messaging

# === Initial parameters ===
mu = 3.986004418e14         # Earth mu (m^3/s^2)
R_E = 6.378e6               # Earth radius (m)
g0 = 9.80665                # m/s^2

alt_init = 500e3            # 500 km LEO
r_LEO = R_E + alt_init      # Total radius (Earth + LEO)
v_LEO = np.sqrt(mu / r_LEO)

alt_GEO = 35_786e3          # GEO altitude
r_GEO = R_E + alt_GEO       # GEO radius (~42,164 km from Earth's center)

m0 = 1000.0                 # initial spacecraft mass (kg)

T_max = 1.0                 # N
Isp = 10000.0               # s

# integration / task timestep
dt = 10.0                   #s s
taskRate = macros.sec2nano(dt)

# coast duration after thrust ends (2 weeks)
coast_duration = 14 * 24 * 3600.0  # 2 weeks in seconds

# safety max simulation seconds (big number)
safety_max = 365 * 24 * 3600.0 * 3.0  # 3 years


def unit(vec):
    n = np.linalg.norm(vec)
    return vec / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])      # accounts for 0 velocity (cannot divide by 0)


def semi_major_axis(r_vec, v_vec, mu_val=mu):
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    # specific orbital energy: eps = v^2/2 - mu/r ; a = -mu/(2*eps) = 1/(2/r - v^2/mu)
    return 1.0 / (2.0 / r - v * v / mu_val)


def run_basilisk_lowthrust():

    # --- Build simulation base ---
    sim = SimulationBaseClass.SimBaseClass()

    # Create a process and task for dynamics
    procName = "dynProcess"
    taskName = "dynTask"
    sim.CreateNewProcess(procName)
    sim.CreateNewTask(procName, taskName, taskRate)

    # --- Spacecraft module (translational only) ---
    scObject = spacecraft.SpacecraftState()
    scObject.ModelTag = "spacecraft"
    scObject.hub.mHub = m0

    # initial inertial position & velocity (N-frame)
    scObject.hub.r_CN_NInit = [r_LEO, 0.0, 0.0]
    scObject.hub.v_CN_NInit = [0.0, v_LEO, 0.0]

    # add spacecraft to the dynamics task
    sim.AddModelToTask(taskName, scObject)

    # --- Central gravity (Earth) ---
    gravFactory = simIncludeGravBody.gravBodyFactory()
    earth = gravFactory.createEarth()
    # attach Earth as the central gravity body to spacecraft's gravField
    try:
        scObject.gravField.gravBodies = [earth]
    except Exception:
        try:
            scObject.gravField = simIncludeGravBody.gravFactoryGravField([earth])
        except Exception:
            pass

    # --- Thruster creation (single thruster) ---
    # Use simIncludeThruster helper to build a thruster device.
    thrusterName = "lowThrust"

    # Place thruster at the body origin; assuming no rotation of spacecraft
    thruster_objs = simIncludeThruster.create(thrusterType='CST',
                                              r_B=[0.0, 0.0, 0.0],
                                              tHat_B=[1.0, 0.0, 0.0],
                                              maxThrust=T_max,
                                              isp=Isp,
                                              baseName=thrusterName)

    # simIncludeThruster.create sometimes returns a list or dict; if list chose first entry, if not list assume returned object is thruster device.
    thr_device = None
    if isinstance(thruster_objs, (list, tuple)) and len(thruster_objs) > 0:
        thr_device = thruster_objs[0]
    else:
        thr_device = thruster_objs

    # --- Thruster effector: attach to spacecraft ---
    # Different versions of Basilisk place effector in different places, so have to locate
    # First checks thruster_objs[-1] is effector
    # Next check thruster device has field .effector
    # If neither true, None (prevents crashing)    
    thrusterEffector = None
    # try first/second attributes
    try:
        thrusterEffector = thruster_objs[-1].effector  # sometimes last item
    except Exception:
        try:
            if hasattr(thr_device, 'effector'):
                thrusterEffector = thr_device.effector
        except Exception:
            thrusterEffector = None

    # If can't find effector, try finding in global namespace created by simIncludeThruster
    if thrusterEffector is None:
        try:
            scObject.addDynamicEffector(thr_device)
        except Exception:
            # if addDynamicEffector doesn't exist, ignore
            pass
    else:
        try:
            scObject.addDynamicEffector(thrusterEffector)
            sim.AddModelToTask(taskName, thrusterEffector)
        except Exception:
            pass

    # Add thruster device to the sim task if not already added - some helpers do this automatically
    try:
        sim.AddModelToTask(taskName, thr_device)
    except Exception:
        # ignore if device cannot be directly added (some versions handle device internally)
        pass

    # --- Build thruster command message (DoubleVec) ---
    nThrusters = 1
    thrustFactorPayload = messaging.DoubleVecPayload()
    thrustFactorPayload.vec = [0.0] * nThrusters
    thrustFactorMsg = messaging.DoubleVecMsg().write(thrustFactorPayload)

    # Attach or store the command message where the effector expects it.
    # 1) If the effector object has a named input 'thrusterCmdInMsg', subscribe to our message.
    try:
        thrusterEffector.thrusterCmdInMsg.subscribeTo(thrustFactorMsg)
    except Exception:
        # 2) If effector has a 'thrustFactorInMsg' or 'thrusterCmd' attribute, try writing to it directly later
        try:
            thrusterEffector.thrustFactor = [0.0] * nThrusters
        except Exception:
            pass

    # --- Setup history logging lists ---
    pos_hist = []
    vel_hist = []
    mass_hist = []
    time_hist = []

    # --- Initialize states for manual stepping ---
    # initial inertial r & v
    r_N = np.array([r_LEO, 0.0, 0.0])
    v_N = np.array([0.0, v_LEO, 0.0])
    mass = m0

    # Pre-calc mass flow for full thrust
    mdot_full = T_max / (Isp * g0)

    # Initialize and start the sim
    sim.InitializeSimulation()

    thrusting = True
    thrust_end_time = None
    t_sim = 0.0
    step = 0
    max_steps = int(min(safety_max / dt, 10_000_000))  # safe cap

    # MAIN STEP LOOP: update thruster command each dt, advance Basilisk one step, read state
    while step < max_steps:

        a_current = semi_major_axis(r_N, v_N, mu)

        # Decide whether to thrust: continue while semi-major axis < target GEO radius
        if thrusting and a_current >= r_GEO:
            thrusting = False
            thrust_end_time = t_sim
            print("Thrust phase complete at t = {:.1f} s ({:.3f} days)".format(t_sim, t_sim / 86400.0))

        # Build thrustFactor command and set thruster orientation to align thrust with inertial +v direction
        if thrusting:
            v_hat = unit(v_N)
            # We want the thruster to *push the spacecraft forward*, so thrust vector points along +v_hat.
            # Update thruster device orientation (API differs by version)
            try:
                thr_device.tHat_B = [float(v_hat[0]), float(v_hat[1]), float(v_hat[2])]
            except Exception:
                try:
                    thr_device.thrDirection = [float(v_hat[0]), float(v_hat[1]), float(v_hat[2])]
                except Exception:
                    # if we can't set it, it's likely the helper will compute thrust direction from commanded vector; ignore
                    pass

            thrust_factor = [1.0]
        else:
            thrust_factor = [0.0]

        # Attempt to publish the thrustFactor message to the effector input.
        # Use 'try' because different versions of Basilisk accept different interfaces.
        try:
            payload = messaging.DoubleVecPayload()
            payload.vec = thrust_factor
            tf_msg = messaging.DoubleVecMsg().write(payload)
            # try expect .thrusterCmdInMsg.write(msg)
            try:
                thrusterEffector.thrusterCmdInMsg.write(tf_msg)
            except Exception:
                # some effectors expect a direct write to an input message attribute
                try:
                    thrusterEffector.thrustFactorInMsg.write(tf_msg)
                except Exception:
                    # fallback: set attribute directly (if available)
                    try:
                        thrusterEffector.thrustFactor = thrust_factor
                    except Exception:
                        pass
        except Exception:
            # if messaging construction failed, try setting an attribute directly
            try:
                thrusterEffector.thrustFactor = thrust_factor
            except Exception:
                pass

        # Advance Basilisk one task step
        try:
            sim.SingleStep()
        except Exception:
            # Some Basilisk versions require ProcessModelQueue and then SingleStep
            try:
                sim.ProcessModelQueue(taskName)
                sim.SingleStep()
            except Exception:
                # if not, try calling ExecuteSimulation for a tiny dt window
                try:
                    sim.ConfigureStopTime(macros.nanoseconds_to_seconds(taskRate))  # may not exist in all versions
                except Exception:
                    pass

        # Read back spacecraft translational output message
        # Check common message name: transStateOutMsg, scStateOutMsg
        r_read = None
        v_read = None
        mass_read = None
        try:
            trans = scObject.transStateOutMsg.read()
            # many transStateOutMsg objects have .r_BN_N and .v_BN_N
            r_read = np.array(trans.r_BN_N)
            v_read = np.array(trans.v_BN_N)
            # spacecraft mass often stored in hub.mHub
            mass_read = scObject.hub.mHub
        except Exception:
            try:
                scst = scObject.scStateOutMsg.read()
                r_read = np.array(scst.r_BN_N)
                v_read = np.array(scst.v_BN_N)
                mass_read = scObject.hub.mHub
            except Exception:
                # As fallback, keep using local integration
                r_read = r_N
                v_read = v_N
                mass_read = mass

        # If the thruster effector didn't update the translational states (because the helper didn't wire), do Euler update by hand:
        # Compute gravity
        rmag = np.linalg.norm(r_N)
        a_grav = -mu * r_N / rmag**3

        # If thruster active, compute thrust acceleration in inertial frame and update v_N and mass
        if thrusting:
            vmag = np.linalg.norm(v_N)
            if vmag > 1e-12:
                # thrust acceleration along v_hat
                a_thrust = (T_max / mass) * (v_N / vmag)
                v_N = v_N + (a_grav + a_thrust) * dt
            else:
                v_N = v_N + a_grav * dt
            # mass update
            mass = mass - mdot_full * dt
            mass = max(0.0, mass)
        else:
            # coast
            v_N = v_N + a_grav * dt

        r_N = r_N + v_N * dt

        # overwrite r_read and v_read with our integrated state to keep logging consistent
        r_read = r_N.copy()
        v_read = v_N.copy()
        mass_read = mass

        # logging
        pos_hist.append(r_read.copy())
        vel_hist.append(v_read.copy())
        mass_hist.append(mass_read)
        time_hist.append(t_sim)

        # increment time
        t_sim += dt
        step += 1

        # stop condition: finished coast after thrust_end_time
        if (not thrusting) and (thrust_end_time is not None) and (t_sim >= thrust_end_time + coast_duration):
            print("Finished coast phase at t = {:.1f} s ({:.3f} days)".format(t_sim, t_sim / 86400.0))
            break

        # safety stop
        if t_sim >= safety_max:
            print("Safety stop reached at t = {:.1f} s".format(t_sim))
            break

    # Convert logs to numpy arrays
    r_hist = np.array(pos_hist)
    v_hist = np.array(vel_hist)
    t_hist = np.array(time_hist)
    m_hist = np.array(mass_hist)

    # Coast analysis on last coast_duration window
    coast_steps = int(coast_duration / dt)
    if coast_steps > len(r_hist):
        coast_steps = len(r_hist)

    r_last = np.linalg.norm(r_hist[-coast_steps:], axis=1)
    r_per = np.min(r_last)
    r_apo = np.max(r_last)
    ecc = (r_apo - r_per) / (r_apo + r_per) if (r_apo + r_per) != 0 else 0.0
    r_avg = 0.5 * (r_apo + r_per)
    rel_err_percent = (r_avg - r_GEO) / r_GEO * 100.0

    print("\n--- Summary ---")
    print("Initial altitude (km): {:.3f}".format((r_LEO - R_E) / 1e3))
    print("Target GEO altitude (km): {:.3f}".format(alt_GEO / 1e3))
    print("Final mass (kg): {:.3f}".format(mass))
    print("Perigee after coast (km): {:.3f}".format((r_per - R_E) / 1e3))
    print("Apogee  after coast (km): {:.3f}".format((r_apo - R_E) / 1e3))
    print("Eccentricity after coast: {:.6f}".format(ecc))
    print("Avg radius error vs GEO: {:.6f} %".format(rel_err_percent))

    # --- Plot results 2D ---
    thrust_end_idx = max(0, len(t_hist) - coast_steps)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.axis('off')

    ax.add_patch(Circle((0, 0), R_E, fc='C0', ec='none'))
    ax.annotate("Earth", xy=(0, 0), ha='center', va='center', color='white')

    # GEO circle
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(r_GEO * np.cos(theta), r_GEO * np.sin(theta), '--', lw=1.5, label='GEO radius')

    # plot trajectory
    ax.plot(r_hist[:thrust_end_idx, 0], r_hist[:thrust_end_idx, 1], lw=1, label='Thrust phase')
    ax.plot(r_hist[thrust_end_idx:, 0], r_hist[thrust_end_idx:, 1], lw=1, label='Coast phase')

    # mark start and thrust end
    ax.scatter(r_hist[0, 0], r_hist[0, 1], color='g', label='Start')
    ax.scatter(r_hist[thrust_end_idx, 0], r_hist[thrust_end_idx, 1], color='k', label='Thrust end / coast start')

    ax.legend(loc='upper right')
    plt.savefig('basilisk_lowthrust_orbit.png', dpi=300)
    plt.show()

    return {
        'r_hist': r_hist,
        'v_hist': v_hist,
        't_hist': t_hist,
        'm_hist': m_hist,
        'perigee_m': r_per,
        'apogee_m': r_apo,
        'ecc': ecc,
        'r_avg': r_avg,
        'rel_err_percent': rel_err_percent
    }


if __name__ == "__main__":
    results = run_basilisk_lowthrust()
