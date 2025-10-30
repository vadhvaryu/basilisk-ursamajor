import os
import numpy as np
import matplotlib.pyplot as plt

from Basilisk.utilities import (SimulationBaseClass, macros, orbitalMotion,
                                simIncludeGravBody, unitTestSupport)
from Basilisk.simulation import spacecraft
from Basilisk import __path__

bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])


def run(show_plots=True):
    scSim = SimulationBaseClass.SimBaseClass()
    scSim.SetProgressBar(True)

    simProcessName = "SimProcess"
    simTaskName = "SimTask"
    simulationTimeStep = macros.sec2nano(1.0)
    dynProcess = scSim.CreateNewProcess(simProcessName)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "CubeSat"
    scSim.AddModelToTask(simTaskName, scObject)

    gravFactory = simIncludeGravBody.gravBodyFactory()
    planet = gravFactory.createEarth()
    planet.isCentralBody = True
    mu = planet.mu
    gravFactory.addBodiesTo(scObject)

    altitude = 500.0 * 1000.0
    r_equatorial = orbitalMotion.REQ_EARTH + altitude

    oe = orbitalMotion.ClassicElements()
    oe.a = r_equatorial
    oe.e = 0.0
    oe.i = 0.0 * macros.D2R
    oe.Omega = 0.0 * macros.D2R
    oe.omega = 0.0 * macros.D2R
    oe.f = 0.0 * macros.D2R

    rN, vN = orbitalMotion.elem2rv(mu, oe)
    scObject.hub.r_CN_NInit = rN
    scObject.hub.v_CN_NInit = vN

    orbitPeriod = 2 * np.pi / np.sqrt(mu / oe.a ** 3)
    simStopTime = macros.sec2nano(orbitPeriod)

    numPoints = 200
    samplingTime = unitTestSupport.samplingTime(simStopTime, simulationTimeStep, numPoints)
    dataRec = scObject.scStateOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(simTaskName, dataRec)

    scSim.InitializeSimulation()
    scSim.ConfigureStopTime(simStopTime)
    scSim.ExecuteSimulation()

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
