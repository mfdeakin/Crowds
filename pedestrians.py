#!/usr/bin/python3

from math import pi, exp, isinf, copysign
from copy import deepcopy
import numpy as np

class Pedestrian:
    """A pedestrian interface for different energy based motion planners"""
    
    def __init__(self, pos, vel = np.array([0, 0]),
                 radius = 1.0, **kwds):
        super().__init__(**kwds)
        self.pos = np.array(pos)
        self.vel = np.array(vel)
        self.radius = radius
    
    def __repr__(self):
        return self.pedType() + "\n" + \
            "Position: " + str(self.pos) + \
            "\nVelocity: " + str(self.vel) + \
            "\nRadius: " + str(self.radius)
    
    def __str__(self):
        return self.__repr__()
    
    def calcPedForce(self, other):
        return np.array([0.0, 0.0])
    
    def calcWallForce(self, wall):
        return np.array([0.0, 0.0])
    
    def update(self, others, walls, dt = 0.1):
        return np.array([0.0, 0.0])
    
    def pedType(self):
        return "Static Pedestrian"

class PedestrianGoal(Pedestrian):
    def __init__(self, goal, color, maxVelMag = 0.1, **kwds):
        super().__init__(**kwds)
        self.goal = goal
        self.maxVelMag = maxVelMag
        self.color = color

    def __repr__(self):
        return super().__repr__() + \
            "\n" + str(self.goal)
    
    def velForce(self):
        vMag = np.sqrt(np.dot(self.vel, self.vel))
        coeff = vMag / 2 / self.maxVelMag
        velForce = -coeff * self.vel
        return velForce
    
    def calcForces(self, others, walls):
        force = np.array([0.0, 0.0])
        for p in others:
            force += self.calcPedForce(p)
        for w in walls:
            force += self.calcWallForce(w)
        force += self.velForce()
        force += self.goal.calcForceToPed(self)
        return force
    
    def update(self, others, walls, dt = 0.1):
        force = self.calcForces(others, walls)
        dv = force * dt
        # Update the velocity
        self.vel = self.vel + dv
        # Impose a hard limit on the velocity
        velMag = np.dot(self.vel, self.vel)
        if velMag > self.maxVelMag ** 2:
            self.vel = self.vel / np.sqrt(velMag) * self.maxVelMag
        # Update the position
        self.pos += self.vel * dt
        return dv

class PedestrianInvDistance(PedestrianGoal):
    """A pedestrian based off of the inverse square distance model"""
    dist_const = 1
    
    def __init__(self, dist_const, **kwds):
        self.dist_const = dist_const
        super().__init__(**kwds)
    
    def calcPedForce(self, other):
        dp = self.pos - other.pos
        magnitude = self.dist_const / np.dot(dp, dp)
        direction = dp * np.sqrt(magnitude)
        return magnitude * direction
    
    def calcWallForce(self, wall):
        dx = wall[0][0] - wall[1][0]
        dy = wall[0][1] - wall[1][1]
        perpVec = np.array([dy, -dx])
        end1 = wall[0] - self.pos
        end2 = wall[1] - self.pos
        sign1 = copysign(1, np.cross(perpVec, end1))
        sign2 = copysign(1, np.cross(perpVec, end2))
        force = np.array([0.0, 0.0])
        if sign1 != sign2:
            # The perpendicular from the wall to the pedestrian exists
            # So, compute the perpendicular distance
            pMag = np.sqrt(np.dot(perpVec, perpVec))
            dist = np.dot(perpVec, end1) / pMag
            force = -self.dist_const * perpVec / dist ** 3
        else:
            # Compute the closest wall end point,
            # and treat it as a pedestrian of 0 radius
            closest = wall[0]
            if np.dot(end1, end1) > np.dot(end2, end2):
                closest = wall[1]
            ped = Pedestrian(closest)
            force = self.calcPedForce(ped)
        return force
    
    def pedType(self):
        return "Inverse Distance Pedestrian"

class PedestrianTTC(PedestrianGoal):
    """A pedestrian based off of the time to collision model"""
    k_const = 1
    tau0 = 3
    
    def __init__(self, **kwds):
        super().__init__(**kwds)
    
    def calcPedTimeToCollision(self, other):
        dv = other.vel - self.vel
        dvMag = np.sqrt(np.dot(dv, dv))
        dp = other.pos - self.pos
        dpMag = np.sqrt(np.dot(dp, dp))
        a = dvMag * dvMag
        b = -np.dot(dp, dv)
        c = dpMag * dpMag - (self.radius + other.radius) ** 2
        d = b * b - a * c
        if a == 0 or d < 0:
            return float('inf')
        else:
            ttc = (b - np.sqrt(d)) / a
            return ttc
    
    def calcWallTimeToCollision(self, wall):
        # Use law of sines to compute the distance to the wall
        end1 = wall[0] - self.pos
        end2 = wall[1] - self.pos
        dw = wall[0] - wall[1]
        dwMag = np.sqrt(np.dot(dw, dw))
        end1Mag = np.sqrt(np.dot(end1, end1))
        phi = np.arccos(np.dot(dw, end1) / dwMag / end1Mag)
        velMag = np.sqrt(np.dot(self.vel, self.vel))
        theta = np.arccos(np.dot(self.vel, end1) / velMag / end1Mag)
        gamma = np.pi - theta - phi
        centralDist = np.sin(phi) * end1Mag / np.sin(gamma)
        # Now find the distance to the actual intersection
        radialDist = centralDist - self.radius / np.sin(gamma)
        # Verify that this occurs where a
        # perpendicular to the wall exists
        intersectPos = self.pos + self.vel / velMag * radialDist
        perpDir = np.array([dw[1], -dw[0]])
        interEnd1 = wall[0] - intersectPos
        interEnd2 = wall[1] - intersectPos
        sign1 = copysign(1, np.cross(interEnd1, perpDir))
        sign2 = copysign(1, np.cross(interEnd2, perpDir))
        ttc = float('inf')
        if sign1 == sign2:
            # The perpendicular does not exist,
            # so compute the TTC for both of the end points
            # Choose the smaller one, as that will be the TTC
            # for the wall
            ped1 = Pedestrian(pos = wall[0])
            t1 = self.calcPedTimeToCollision(ped1)
            ped2 = Pedestrian(pos = wall[1])
            t2 = self.calcPedTimeToCollision(ped2)
            times = sorted([t1, t2])
            if times[0] > 0:
                ttc = times[0]
            elif times[1] > 0:
                ttc = times[1]
            else:
                ttc = float('inf')
        else:
            # The perpendicular does exist,
            # so just compute the time to this position
            ttc = radialDist / vel.mag
        return ttc
    
    def calcEnergy(self, other):
        ttc = self.calcTimeToCollision(other)
        e = self.k_const * exp(-ttc / self.tau0) / ttc ** 2
        return e
    
    def calcPedForce(self, other):
        ttc = self.calcPedTimeToCollision(other)
        if isinf(ttc) or ttc < 0:
            return np.array([0.0, 0.0])
        dp = other.pos - self.pos
        dv = other.vel - self.vel
        dpMag = np.sqrt(np.dot(dp, dp))
        dvMag = np.sqrt(np.dot(dv, dv))
        dpdv = np.dot(dp, dv)
        totalRad = self.radius + other.radius
        fterm1 = -self.k_const * exp(-ttc / self.tau0) / ((dpMag * ttc)  ** 2)
        fterm2 = 2 / ttc + 1 / self.tau0
        fterm3num = (dvMag ** 2) * dp - dpdv * dv
        fterm3den = dpdv ** 2 - (dvMag ** 2) * (dpMag ** 2 - totalRad ** 2)
        force = -fterm1 * fterm2 * (dv - fterm3num / np.sqrt(fterm3den))
        return force
    
    def calcWallForce(self, wall):
        force = np.array([0.0, 0.0])
        return force
    
    def pedType(self):
        return "Time To Collision Pedestrian"

class PedestrianDS(PedestrianGoal):
    def __init__(self, safeDist = 0.2, springConst = 1,
                 dampConst = 1, **kwds):
        super().__init__(**kwds)
        self.safeDist = safeDist
        self.springConst = springConst
        self.dampConst = dampConst
    
    def calcPedForce(self, other):
        dp = self.pos - other.pos
        dpMag = np.sqrt(np.dot(dp, dp))
        if dpMag > safeDist:
            return 0
        dv = self.vel - other.vel
        dpDir = dp / dpMag
        compSpeed = np.dot(dv, dpDir)
        compression = safeDist - dpMag
        force = (springConst * compression - dampConst * compSpeed) * dpDir
        return force
    
    def pedType(self):
        return "Damped Spring Pedestrian"
