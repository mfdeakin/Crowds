
import numpy as np

class Goal:
    def __init__(self, pos, forceCoeff = 1.0, **kwds):
        super().__init__(**kwds)
        self.pos = np.array(pos)
        self.force = forceCoeff
    
    def __repr__(self):
        return "Goal Type: " + self.goalType() + "\n" + \
            "Position: " + str(self.pos) + "\n" + \
            "Force Coefficient: " + str(self.force)
    
    def __str__(self):
        return self.__repr__()
    
    @staticmethod
    def goalType():
        return "CForce"
    
    def calcForceToPed(self, ped):
        disp = ped.pos - self.pos
        forceDir = -disp / np.sqrt(np.dot(disp, disp))
        force = forceDir * self.force
        return force

class DistanceGoal(Goal):
    def __init__(self, **kwds):
        super().__init__(**kwds)
    
    @staticmethod
    def goalType():
        return "DForce"
    
    def calcForceToPed(self, ped):
        disp = self.pos - ped.pos
        force = disp * self.force
        return force

class TimeToGoal(Goal):
    def __init__(self, distCoeff = 1.0, timeCoeff = 0.1, **kwds):
        self.distCoeff = distCoeff
        self.timeCoeff = timeCoeff
        super().__init__(**kwds)
    
    @staticmethod
    def goalType():
        return "TTGForce"
    
    def calcTimeToMinDist(self, ped):
        dv = ped.vel
        dp = self.pos - ped.pos
        dvMag = np.sqrt(np.dot(dv, dv))
        if dvMag == 0:
            minDist = np.sqrt(np.dot(dp, dp))
            # The pedestrian is already at the minimum distance,
            # so it doesn't make sense to have a force associated
            # with the time to reach it
            return (minDist, 0)
        dvDir = dv / dvMag
        ptMinDist = np.dot(dvDir, dp)
        minDistTime = ptMinDist / dvMag
        minDistVec = dp - ptMinDist * dvDir
        minDist = np.sqrt(np.dot(minDistVec, minDistVec))
        return (minDist, minDistTime)
    
    def calcForceToPed(self, ped):
        minDist, minDistTime = self.calcTimeToMinDist(ped)
        dp = self.pos - ped.pos
        dpDir = dp / np.sqrt(np.dot(dp, dp))
        if minDistTime < 0:
            minDistTime *= -2
        f = self.distCoeff * minDist + \
            self.timeCoeff * minDistTime
        force = dpDir * f
        return force

