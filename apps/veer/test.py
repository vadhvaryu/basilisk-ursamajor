from Basilisk.utilities.SimulationBaseClass import SimBaseClass
from Basilisk.utilities import simIncludeGravBody

sim = SimBaseClass()
step_us = int(1.0 * 1e6)
dynProcess = sim.CreateNewProcess("dynProcess")
dynTask = sim.CreateNewTask("dynTask", step_us, 0)
dynProcess.addTask(dynTask)

gravFactory = simIncludeGravBody.gravBodyFactory()
earth = gravFactory.createCustomGravObject("earth", 3.986004418e14, radEquator=6378137.0)
earth.isCentralBody = True

print("✅ grav body created successfully:", earth)
