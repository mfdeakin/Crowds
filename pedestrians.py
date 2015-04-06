#!/usr/bin/python3

from math import sqrt, pi, exp, isinf
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
    
    def calcForce(self, other):
        return np.array([0.0, 0.0])
    
    def update(self, others, dt = 0.1):
        dv = np.array([0.0, 0.0])
        for p in others:
            force = self.calcForce(p)
            print()
            print(self)
            print(p)
            print(force)
            dv += force * dt
        dv += self.goal.calcForceToPed(self) * dt
        self.vel += dv
        self.pos += self.vel * dt
        return dv
    
    def pedType(self):
        return "Static Pedestrian"

class PedestrianGoal(Pedestrian):
    def __init__(self, goal, **kwds):
        super().__init__(**kwds)
        self.goal = goal

    def __repr__(self):
        return super().__repr__() + \
            "\n" + str(self.goal)

class PedestrianInvDistance(PedestrianGoal):
    """A pedestrian based off of the inverse square distance model"""
    dist_const = 1
    
    def __init__(self, dist_const, **kwds):
        self.dist_const = dist_const
        super().__init__(kwds)
    
    def calcForce(self, other):
        dp = self.pos - other.pos
        magnitude = self.dist_const / np.dot(dp, dp)
        direction = dp * sqrt(magnitude)
        return magnitude * direction
    
    def pedType(self):
        return "Inverse Distance Pedestrian"

class PedestrianTTC(PedestrianGoal):
    """A pedestrian based off of the time to collision model"""
    k_const = 1
    tau0 = 3
    
    def __init__(self, **kwds):
        super().__init__(**kwds)
    
    def calcTimeToCollision(self, other):
        dv = other.vel - self.vel
        dvMag = sqrt(np.dot(dv, dv))
        dp = other.pos - self.pos
        dpMag = sqrt(np.dot(dp, dp))
        a = dvMag * dvMag
        b = -np.dot(dp, dv)
        c = dpMag * dpMag - (self.radius + other.radius) ** 2
        d = b * b - a * c
        if a == 0 or d < 0:
            return float('inf')
        else:
            ttc = (b - sqrt(d)) / a
            return ttc
    
    def calcEnergy(self, other):
        ttc = self.calcTimeToCollision(other)
        e = self.k_const * exp(-ttc / self.tau0) / ttc ** 2
        return e
    
    def calcForce(self, other):
        ttc = self.calcTimeToCollision(other)
        if isinf(ttc):
            return np.array([0.0, 0.0])
        dp = other.pos - self.pos
        dv = other.vel - self.vel
        dpMag = sqrt(np.dot(dp, dp))
        dvMag = sqrt(np.dot(dv, dv))
        dpdv = np.dot(dp, dv)
        totalRad = self.radius + other.radius
        fterm1 = -self.k_const * exp(-ttc / self.tau0) / ((dpMag * ttc)  ** 2)
        fterm2 = 2 / ttc + 1 / self.tau0
        fterm3num = (dvMag ** 2) * dp - dpdv * dv
        fterm3den = dpdv ** 2 - (dvMag ** 2) * (dpMag ** 2 - totalRad ** 2)
        force = -fterm1 * fterm2 * (dv - fterm3num / sqrt(fterm3den))
        return force
    
    def pedType(self):
        return "Time To Collision Pedestrian"

class PedestrianDS(PedestrianGoal):
    safeDist = 3
    springConst = 1
    dampConst = 1
    
    def calcForce(self, other):
        dp = other.pos - self.pos
        dpMag = sqrt(np.dot(dp, dp))
        if dpMag > safeDist:
            return 0
        dv = other.vel - self.vel
        dvMag = sqrt(np.dot(dv, dv))
        force = springConst * dp / dpMag + dampConst * dv / dvMag
        return force
    
    def pedType(self):
        return "Damped Spring Pedestrian"
