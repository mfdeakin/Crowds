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
    
    def intersectingPed(self, other):
        disp = self.pos - other.pos
        dist = np.dot(disp, disp)
        totalRad = (self.radius + other.radius) ** 2
        return totalRad > dist
    
    def intersectingWall(self, wall):
        wallVec = wall[0] - wall[1]
        wallDir = wallVec / np.sqrt(np.dot(wallVec, wallVec))
        wallPerp = np.array([-wallDir[1], wallDir[0]])
        end1 = wall[0] - self.pos
        if abs(np.dot(wallPerp, end1)) < self.radius:
            # Possible collision, check if the
            # perpendicular intersects the wall
            end2 = wall[1] - self.pos
            sign1 = copysign(1, np.cross(end1, wallPerp))
            sign2 = copysign(1, np.cross(end2, wallPerp))
            if sign1 == sign2:
                # The perpendicular doesn't exist,
                # though there's the possibility
                # it's still intersecting at an end
                if np.dot(end1, end1) < self.radius ** 2 or \
                   np.dot(end2, end2) < self.radius ** 2:
                    return True
            else:
                return True
        return False
    
    @staticmethod
    def pedType(self):
        return "SPed"

class PedestrianGoal(Pedestrian):
    def __init__(self, goal, color, wallCoeff = 0.01,
                 pedCoeff = 1.0, goalCoeff = 1,
                 maxVelMag = 0.1, **kwds):
        super().__init__(**kwds)
        self.goal = goal
        self.maxVelMag = maxVelMag
        self.wallCoeff = wallCoeff
        self.pedCoeff = pedCoeff
        self.goalCoeff = goalCoeff
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
            force += self.calcPedForce(p) * self.pedCoeff
        for w in walls:
            force += self.calcWallForce(w) * self.wallCoeff
        force += self.velForce()
        force += self.goal.calcForceToPed(self) * self.goalCoeff
        return force
    
    def goalReached(self):
        disp = self.pos - self.goal.pos
        dist = np.dot(disp, disp)
        return dist < self.radius ** 2
    
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
        dp = self.pos - other.pos - self.radius - other.radius
        magnitude = self.dist_const / np.dot(dp, dp)
        direction = dp * np.sqrt(magnitude)
        return magnitude * direction
    
    def calcWallForce(self, wall):
        wallVec = wall[0] - wall[1]
        wallDir = wallVec / np.sqrt(np.dot(wallVec, wallVec))
        wallPerp = np.array([-wallDir[1], wallDir[0]])
        end1 = wall[0] - self.pos
        end2 = wall[1] - self.pos
        # Make certain the wall is always in the direction of wallPerp
        if np.dot(wallPerp, end1) < 0:
            wallPerp = -wallPerp
        sign1 = copysign(1, np.cross(wallPerp, end1))
        sign2 = copysign(1, np.cross(wallPerp, end2))
        force = np.array([0.0, 0.0])
        if sign1 != sign2:
            # The perpendicular from the wall to the pedestrian exists
            # So compute the perpendicular distance and force
            dist = np.dot(wallPerp, end1)
            force = -self.dist_const * wallPerp / dist ** 2
        else:
            # Compute the closest wall end point,
            # and treat it as a pedestrian of 0 radius
            closest = wall[0]
            if np.dot(end1, end1) > np.dot(end2, end2):
                closest = wall[1]
            ped = Pedestrian(pos = closest, radius = 0)
            force = self.calcPedForce(ped)
        return force
    
    @staticmethod
    def pedType():
        return "IDPed"

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
        if np.dot(self.vel, self.vel) == 0:
            return float('inf')
        # Compute the time to collision of an infinitely long wall
        wallVec = wall[1] - wall[0]
        wallDir = wallVec / np.sqrt(np.dot(wallVec, wallVec))
        wallPerp = np.array([wallDir[1], -wallDir[0]])
        # Perpendiculars are unsigned, but computations are not
        # It should point from the pedestrian to the wall,
        # not the other way around
        end1 = wall[0] - self.pos
        if np.dot(end1, wallPerp) < 0:
            wallPerp = -wallPerp
        perpDist = np.dot(end1, wallPerp)
        perpVel = abs(np.dot(wallPerp, self.vel))
        if perpDist > 0:
            if perpVel > 0:
                ttc = (perpDist - self.radius) / \
                      abs(np.dot(wallPerp, self.vel))
            else:
                ttc = float('inf')
        else:
            ttc = 0
        return ttc
    
    def isWallCollision(self, ttc, wall):
        # Determine if the perpendicular from the pedestrian
        # at the future time is intersects the wall
        wallVec = wall[1] - wall[0]
        wallDir = wallVec / np.sqrt(np.dot(wallVec, wallVec))
        wallPerp = np.array([wallDir[1], -wallDir[0]])
        # Perpendiculars are unsigned, but computations are not
        # It should point from the pedestrian to the wall
        end1 = wall[0] - self.pos
        if np.dot(end1, wallPerp) < 0:
            wallPerp = -wallPerp
        # collisionPos gives the position of the collision
        # relative to one of the ends of the wall
        collisionPos = ttc * self.vel + self.radius * wallPerp - end1
        collisionProd = np.dot(collisionPos, wallVec)
        wallLen = np.dot(wallVec, wallVec)
        # This tells us if the collision occurs or not
        # If this is larger than the length squared of the wall,
        # it's not a collision
        # If this is less than 0, it's not a collision
        if collisionProd > wallLen or collisionProd < 0:
            return False
        else:
            return True
    
    def calcWallTTCGrad(self, wall):
        wallVec = wall[1] - wall[0]
        wallDir = wallVec / np.sqrt(np.dot(wallVec, wallVec))
        wallPerp = np.array([-wallDir[1], wallDir[0]])
        end1 = wall[0] - self.pos
        if np.dot(end1, wallPerp) < 0:
            wallPerp = -wallPerp
        perpVel = np.dot(self.vel, wallPerp)
        if perpVel > 0:
            return wallPerp / perpVel
        else:
            return np.array([0.0, 0.0])
    
    def calcEnergy(self, other):
        ttc = self.calcPedTimeToCollision(other)
        e = self.k_const * exp(-ttc / self.tau0) / ttc ** 2
        return e
    
    def calcPedTTCGrad(self, other):
        dp = other.pos - self.pos
        dv = other.vel - self.vel
        dvMagSq = np.dot(dv, dv)
        num = dvMagSq * dp - np.dot(dp, dv) * dv
        dpMagSq = np.dot(dp, dp)
        radsSq = (self.radius + other.radius) ** 2
        den = np.dot(dp, dv) ** 2 - dvMagSq * (dpMagSq - radsSq)
        if np.dot(num, num) == 0.0 or den <= 0.0:
            return dv / dvMagSq
        return (dv - num / np.sqrt(den)) / dvMagSq
    
    def calcPedForce(self, other):
        ttc = self.calcPedTimeToCollision(other)
        if isinf(ttc) or ttc <= 0:
            return np.array([0.0, 0.0])
        fterm1 = -self.k_const * exp(-ttc / self.tau0) / ttc ** 2
        fterm2 = 2 / ttc + 1 / self.tau0
        ttcGrad = self.calcPedTTCGrad(other)
        force = -fterm1 * fterm2 * ttcGrad
        return force
    
    def calcWallForce(self, wall):
        ttc = self.calcWallTimeToCollision(wall)
        if self.isWallCollision(ttc, wall):
            ttcGrad = self.calcWallTTCGrad(wall)
            fterm1 = -self.k_const * exp(-ttc / self.tau0) / (ttc ** 2)
            fterm2 = 2 / ttc + 1 / self.tau0
            force = fterm1 * fterm2 * ttcGrad
            return force
        else:
            wpeds = [Pedestrian(pos = wall[0], radius = 0.0),
                     Pedestrian(pos = wall[1], radius = 0.0)]
            times = [self.calcPedTimeToCollision(ped) for ped in wpeds]
            if times[0] > 0 and not isinf(times[0]):
                if times[0] < times[1] or times[1] < 0:
                    return self.calcPedForce(wpeds[0])
            if times[1] > 0 and not isinf(times[1]):
                if times[1] < times[0] or times[0] < 0:
                    return self.calcPedForce(wpeds[1])
            # No collision with the ends of the wall
            return np.array([0.0, 0.0])
    
    @staticmethod
    def pedType():
        return "TTCPed"

class PedestrianDS(PedestrianGoal):
    def __init__(self, safeDist = 0.2, springConst = 1,
                 dampConst = 1, **kwds):
        super().__init__(**kwds)
        self.safeDist = safeDist
        self.springConst = springConst
        self.dampConst = dampConst
    
    def calcPedForce(self, other):
        dp = self.pos - other.pos
        dpMag = np.sqrt(np.dot(dp, dp)) - \
                self.radius - other.radius
        if dpMag > self.safeDist:
            return 0
        dv = self.vel - other.vel
        dpDir = dp / (dpMag + self.radius + other.radius)
        compSpeed = np.dot(dv, dpDir)
        compression = self.safeDist - dpMag
        force = (self.springConst * compression - \
                 self.dampConst * compSpeed) * dpDir
        return force
    
    def calcWallForce(self, wall):
        end1 = wall[0] - self.pos
        wallVec = wall[1] - wall[0]
        wallDir = wallVec / np.sqrt(np.dot(wallVec, wallVec))
        wallPerp = np.array([wallDir[1], -wallDir[0]])
        if np.dot(end1, wallPerp) < 0:
            wallPerp = -wallPerp
        perpDist = np.dot(end1, wallPerp) - self.radius
        if perpDist > self.safeDist:
            return np.array([0.0, 0.0])
        end1Dist = np.dot(wallDir, end1)
        end1Sign = copysign(1, end1Dist)
        end2 = wall[1] - self.pos
        end2Dist = np.dot(wallDir, end2)
        end2Sign = copysign(1, end2Dist)
        if end1Sign != end2Sign:
            xVel = np.dot(self.vel, wallPerp)
            xForce = abs(self.springConst * (end2Dist - end1Dist) * \
                         (self.safeDist - perpDist)) + self.dampConst * xVel
            yForce = (end2Dist ** 2 - end1Dist ** 2) / 2
            force = -xForce * wallPerp - yForce * wallDir
        else:
            pos = wall[0]
            if end1Dist > end2Dist:
                pos = wall[1]
            wallPed = Pedestrian(pos = pos, radius = 0)
            wallLen = abs(end2Dist - end1Dist)
            force = self.calcPedForce(wallPed)
            force *= wallLen
        return force
    
    @staticmethod
    def pedType():
        return "DSPed"
