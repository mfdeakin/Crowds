
import numpy as np
from pedestrians import *
from goals import *

epsilon = 1e-6
wall = (np.array([0.5, 1.0]), np.array([0.5, -1.0]))
wallTopPed = Pedestrian(pos = wall[0], radius = 0.0)
wallBotPed = Pedestrian(pos = wall[1], radius = 0.0)
wallVec = wall[0] - wall[1]
wallDir = wallVec / np.sqrt(np.dot(wallVec, wallVec))
wallPerp = np.array([-wallDir[1], wallDir[0]])
wallLen = np.dot(wallVec, wallDir)
goal = Goal(pos = [1.0, 0.0])
color = "#ff0000"
pedRad = 1 / 2 ** 6

pttc1 = PedestrianTTC(pos = [0.0, 0.0], goal = goal,
                      vel = [0.0625, 0.0], color = color,
                      radius = pedRad)

pttc2 = PedestrianTTC(pos = [0.5, 1.5], goal = goal,
                      vel = [0.0, -0.0625 * 3], color = color,
                      radius = pedRad)

pds1 = PedestrianDS(pos = [0.0, 0.0], goal = goal,
                    safeDist = 1.0,
                    color = color, radius = pedRad)

pds2 = PedestrianDS(pos = [0.5, 1.5], goal = goal,
                    safeDist = 2.0,
                    color = color, radius = pedRad)

pds2 = PedestrianDS(pos = [0.5, 1.5], goal = goal,
                    safeDist = 2.0,
                    color = color, radius = pedRad)

expectedTTC1 = (wall[0][0] - pttc1.pos[0] - pttc1.radius) / pttc1.vel[0]
expectedTTCGrad1 = wallPerp / np.dot(pttc1.vel, wallPerp)
expectedForce1 = -pttc1.k_const * exp(-expectedTTC1 / pttc1.tau0) / expectedTTC1 ** 2 * \
                (2 / expectedTTC1 + 1 / pttc1.tau0) * expectedTTCGrad1

expectedTTC2 = 0
realTTC2 = -(pttc2.pos[1] - wall[0][1] - pttc2.radius) / pttc2.vel[1]
expectedTTCGrad2 = -wallDir / np.dot(pttc2.vel, wallDir)
expectedForce2 = pttc2.k_const * exp(-realTTC2 / pttc2.tau0) / realTTC2 ** 2 * \
                (2 / realTTC2 + 1 / pttc2.tau0) * expectedTTCGrad2

dp = pttc1.pos - pttc2.pos
dv = pttc1.vel - pttc2.vel
dpMag = np.sqrt(np.dot(dp, dp)) - pttc1.radius - pttc2.radius
dvMag = np.sqrt(np.dot(dv, dv))
expectedTTC12 = dpMag / dvMag
expectedTTCGrad12 = np.array([0.0, 0.0])
expectedForce12 = np.array([0.0, 0.0])

expectedDSForce1 = pds1.springConst * (pds1.safeDist - (wall[0][0] - pds1.pos[0] - pds1.radius)) * wallLen * wallPerp
expectedDSForce2 = pds2.springConst * (pds2.safeDist - (pds2.pos[1] - wall[0][1] - pds1.radius)) * wallLen * wallDir

tests = [("Time To Collision of the wall 1",
          pttc1.calcWallTimeToCollision, {'wall': wall},
          expectedTTC1,
          lambda x, y: abs(x - y) < epsilon),
         ("Gradient of TTC for the wall 1",
          pttc1.calcWallTTCGrad, {'wall': wall},
          expectedTTCGrad1,
          lambda x, y: (x == y).all()),
         ("Force from the wall 1",
          pttc1.calcWallForce, {'wall': wall},
          expectedForce1,
          lambda x, y: (x == y).all()),
         
         ("Time To Collision of the wall 2",
          pttc2.calcWallTimeToCollision, {'wall': wall},
          expectedTTC2,
          lambda x, y: abs(x - y) < epsilon),
         ("Gradient of TTC for the wall 2",
          pttc2.calcPedTTCGrad, {'other': wallTopPed},
          expectedTTCGrad2,
          lambda x, y: (x == y).all()),
         ("Force from the wall 2",
          pttc2.calcWallForce, {'wall': wall},
          expectedForce2,
          lambda x, y: (x == y).all()),
         
         ("Time To Collision of peds 1 and 2",
          pttc1.calcPedTimeToCollision, {'other': pttc2},
          expectedTTC12,
          lambda x, y: abs(x - y) < epsilon),
         ("Gradient of TTC for peds 1 and 2",
          pttc1.calcPedTTCGrad, {'other': pttc2},
          expectedTTCGrad12,
          lambda x, y: (x == y).all()),
         ("Force from ped 1 to 2",
          pttc1.calcPedForce, {'other': pttc2},
          expectedForce12,
          lambda x, y: (x == y).all()),
         
         ("Force from the wall DS 1",
          pds1.calcWallForce, {'wall': wall},
          expectedDSForce1,
          lambda x, y: (x == y).all()),
         
         ("Force from the wall DS 2",
          pds2.calcWallForce, {'wall': wall},
          expectedDSForce2,
          lambda x, y: (x == y).all())
]

for t in tests:
    print("Running Test: " + t[0] + ", expecting " + str(t[3]))
    ret = t[1](**t[2])
    if t[4](ret, t[3]):
        print("PASS")
    else:
        print("FAIL: " + str(ret))
