#!/usr/bin/python3

from pedestrians import *
from goals import *
from math import *

import numpy as np
import random
import cairo

class CrowdSim:
    """A pedestrian interface for different energy based motion planners"""
    
    def __init__(self, numPedestrians = 3,
                 pedRadius = 0.05, areaDim = np.array([1.0, 1.0])):
        self.pedestrians = []
        self.walls = []
        self.wallDist = 0.001
        self.pedRadius = pedRadius
        for i in range(numPedestrians):
            xGoal = random.random() * areaDim[0]
            yGoal = random.random() * areaDim[1]
            goal = TimeToGoal(pos = [xGoal, yGoal])
            xPed = random.random() * areaDim[0]
            yPed = random.random() * areaDim[1]
            self.pedestrians.append(PedestrianTTC(pos = [xPed, yPed],
                                                  radius = pedRadius,
                                                  goal = goal))
        for c1 in [np.array([0.0, 0.0]),
                   np.array([areaDim[0], areaDim[1]])]:
            for c2 in [np.array([areaDim[0], 0.0]),
                       np.array([0.0, areaDim[1]])]:
                self.createWall(c1, c2)
    
    def __repr__(self):
        strout = "Number of Pedestrians: " + \
                 str(len(self.pedestrians))
        for p in self.pedestrians:
            strout += "\n" + str(p)
        return strout
    
    def __str__(self):
        return self.__repr__()
    
    def createWall(self, start, end):
        ds = end - start
        dsMag = sqrt(np.dot(ds, ds))
        numPts = int(dsMag / self.wallDist)
        for i in range(numPts):
            pos = start + ds / numPts * i
            wallPed = Pedestrian(pos = pos)
            self.walls.append(wallPed)
    
    def renderScene(self, context):
        context.set_source_rgb(1.0, 1.0, 1.0)
        context.rectangle(0, 0, 1.0, 1.0)
        context.fill()
        context.set_source_rgb(0.0, 0.0, 0.0)
        for w in self.walls:
            context.rectangle(w.pos[0] - self.wallDist / 2,
                              w.pos[1] - self.wallDist / 2,
                              w.pos[0] + self.wallDist / 2,
                              w.pos[1] + self.wallDist / 2)
            context.fill()
        context.set_source_rgb(1.0, 0.0, 0.0)
        for p in self.pedestrians:
            context.arc(p.pos[0], p.pos[1], self.pedRadius, 0, 2 * pi)
    
    def timestep(self):
        for i in range(len(self.pedestrians)):
            otherPeds = self.pedestrians[:i] + self.pedestrians[i + 1:] +\
                        self.walls
            self.pedestrians[i].update(otherPeds)

def createCairoImg(xMax, yMax):
    imScale = 32
    imXDim = imScale * xMax
    imYDim = imScale * yMax
    rmBmp = np.zeros((imXDim, imYDim, 4), dtype=np.uint8)
    rmSurf = cairo.ImageSurface.create_for_data(rmBmp,
                                                cairo.FORMAT_ARGB32,
                                                imXDim, imYDim)
    rmCtx = cairo.Context(rmSurf)
    rmCtx.set_source_rgb(1.0, 1.0, 1.0)
    rmCtx.translate(0, imYDim)
    rmCtx.scale(imScale, -imScale)
    rmCtx.paint()
    rmCtx.set_line_width(0.1)
    return (rmSurf, rmCtx)

if __name__ == "__main__":
    c = CrowdSim()
    print(c)
    for t in np.linspace(0.0, 10.0, 100):
        surface, context = createCairoImg(256, 256)
        c.timestep()
        c.render(context)
        print(c)
    
