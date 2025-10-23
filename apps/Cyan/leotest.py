#
#  ISC License
#  ------------------------------
#  Minimal Basilisk Example:
#  A 3U CubeSat in a 500 km circular equatorial orbit around Earth
#

import os
import numpy as np
import matplotlib.pyplot as plt

# Basilisk core imports
from Basilisk.utilities import (SimulationBaseClass, macros, orbitalMotion,
                                simIncludeGravBody, unitTestSupport)
from Basilisk.simulation import spacecraft
from Basilisk import __path__

bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])


def run(show_plots=True):
    """Run a basic CubeSat orbit simulation around Earth"""

    # ----------------------------------------------------------------------
    # 1️⃣ Create simulation container and process
    # ----------------------------------------------------------------------
    scSim = SimulationBaseClass.SimBaseClass()
    scSim.SetProgressBar(True)                      # optional terminal progress bar

    simProcessName = "SimProcess"
    simTaskName = "SimTask"
    simulationTimeStep = macros.sec2nano(1.0)       # integrate every 1 second
    dynProcess = scSim.CreateNewProcess(simProcessName)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    # ----------------------------------------------------------------------
    # 2️⃣ Create spacecraft object
    # ----------------------------------------------------------------------
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "CubeSat"
    scSim.AddModelToTask(simTaskName, scObject)

    # ----------------------------------------------------------------------
    # 3️⃣ Create gravity body (Earth)
    # ----------------------------------------------------------------------
    gravFactory = simIncludeGravBody.gravBodyFactory()
    planet = gravFactory.createEarth()
    planet.isCentralBody = True                     # Earth is the reference center
    mu = planet.mu

    # Attach gravity to spacecraft
    gravFactory.addBodiesTo(scObject)

    # ----------------------------------------------------------------------
    # 4️⃣ Define orbital elements for 500 km equatorial orbit
    # ----------------------------------------------------------------------
    altitude = 500.0 * 1000.0
    r_equatorial = orbitalMotion.REQ_EARTH + altitude

    oe = orbitalMotion.ClassicElements()
    oe.a = r_equatorial
    oe.e = 0.0                      # circular
    oe.i = 0.0 * macros.D2R         # equatorial
    oe.Omega = 0.0 * macros.D2R     # RAAN
    oe.omega = 0.0 * macros.D2R     # argument of perigee
    oe.f = 0.0 * macros.D2R         # true anomaly

    rN, vN = orbitalMotion.elem2rv(mu, oe)
    scObject.hub.r_CN_NInit = rN
    scObject.hub.v_CN_NInit = vN

    # ----------------------------------------------------------------------
    # 5️⃣ Configure logging
    # ----------------------------------------------------------------------
    orbitPeriod = 2 * np.pi / np.sqrt(mu / oe.a ** 3)
    simStopTime = macros.sec2nano(orbitPeriod)

    numPoints = 200
    samplingTime = unitTestSupport.samplingTime(simStopTime, simulationTimeStep, numPoints)
    dataRec = scObject.scStateOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(simTaskName, dataRec)

    # ----------------------------------------------------------------------
    # 6️⃣ Initialize and run simulation
    # ----------------------------------------------------------------------
    scSim.InitializeSimulation()
    scSim.ConfigureStopTime(simStopTime)
    scSim.ExecuteSimulation()

    # ----------------------------------------------------------------------
    # 7️⃣ Retrieve and plot data
    # ----------------------------------------------------------------------
    posData = dataRec.r_BN_N
    velData = dataRec.v_BN_N
    timeData = dataRec.times() * macros.NANO2SEC

    plt.figure()
    plt.plot(posData[:, 0] / 1000, posData[:, 1] / 1000, label="Orbit")
    plt.xlabel("X [km]")
    plt.ylabel("Y [km]")
    plt.title("3U CubeSat in 500 km Equatorial Orbit")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    if show_plots:
        plt.show()

    return posData, velData, timeData


if __name__ == "__main__":
    run(True)
