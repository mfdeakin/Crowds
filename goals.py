
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
    
    def goalType(self):
        return "Constant Force"
    
    def calcForceToPed(self, ped):
        disp = ped.self.pos - pos
        forceDir = disp / np.sqrt(np.dot(disp, disp))
        force = forceDir * self.force
        return force

class DistanceGoal(Goal):
    def __init__(self, **kwds):
        super().__init__(**kwds)
    
    def goalType(self):
        return "Distance Force"
    
    def calcForceToPed(self, ped):
        disp = self.pos - ped.pos
        force = disp * self.force
        return force

class TimeToGoal(Goal):
    def __init__(self, **kwds):
        super().__init__(**kwds)
    
    def goalType(self):
        return "Time To Goal Force"
    
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
        f = minDist + minDistTime
        force = dpDir * f
        return force

