#!/usr/bin/python3

from pedestrians import *
from goals import *
from math import *

import numpy as np
import random
from PIL import Image, ImageDraw, ImageColor

class CrowdSim:
    """A pedestrian interface for different energy based motion planners"""
    
    def __init__(self, numPedestrians = 3,
                 pedRadius = 0.05,
                 maxVelMag = 0.1,
                 areaDim = np.array([1.0, 1.0])):
        self.pedestrians = []
        self.walls = []
        self.wallDist = 0.001
        self.pedRadius = pedRadius
        colors = ["#ff0000", "#00ff00", "#0000ff"]
        for i in range(numPedestrians):
            xGoal = random.random() * (areaDim[0] - 2 * pedRadius) + \
                    pedRadius
            yGoal = random.random() * (areaDim[1] - 2 * pedRadius) + \
                    pedRadius
            goal = TimeToGoal(pos = [xGoal, yGoal])
            xPed = random.random() * (areaDim[0] - 2 * pedRadius) + \
                   pedRadius
            yPed = random.random() * (areaDim[1] - 2 * pedRadius) + \
                   pedRadius
            print(xPed, yPed, pedRadius)
            color = colors[i % numPedestrians]
            p = PedestrianTTC(pos = [xPed, yPed],
                              radius = pedRadius,
                              goal = goal,
                              maxVelMag = maxVelMag,
                              color = color)
            self.pedestrians.append(p)
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
            wallPed = Pedestrian(pos = pos, radius = 0.0)
            self.walls.append(wallPed)
    
    def renderScene(self, im, width, height):
        draw = ImageDraw.ImageDraw(im)
        draw.setink("#000000")
        for p in self.pedestrians:
            # Draw the pedestrian
            eTopLeft = p.pos - np.array([p.radius, p.radius])
            eBotRight = p.pos - np.array([-p.radius, -p.radius])
            eBounds = (eTopLeft[0] * width, eTopLeft[1] * height,
                       eBotRight[0] * width, eBotRight[1] * height)
            draw.ellipse(eBounds, fill = p.color)
            draw.ellipse(eBounds)
            
            # Draw the pedestrians velocity
            xBounds = [p.pos[0] * width,
                       (p.pos[0] + p.vel[0]) * width]
            yBounds = [p.pos[1] * height,
                       (p.pos[1] + p.vel[1]) * height]
            aBounds = tuple(zip(xBounds, yBounds))
            draw.line(aBounds, width = 1)
    
    def timestep(self):
        for i in range(len(self.pedestrians)):
            otherPeds = self.pedestrians[:i] + self.pedestrians[i + 1:] +\
                        self.walls
            self.pedestrians[i].update(otherPeds)

def createImage(width, height):
    imBytes = np.zeros((width, height, 3))
    im = Image.frombytes(size = (width, height),
                             data = imBytes, mode = "RGB")
    ImageDraw.floodfill(im, (0, 0), (255, 255, 255))
    return im

if __name__ == "__main__":
    c = CrowdSim()
    print(c)
    imWidth = 512
    imHeight = 512
    for t in np.linspace(0.0, 10.0, 101):
        im = createImage(imWidth, imHeight)
        c.renderScene(im, imWidth, imHeight)
        c.timestep()
        im.save("Time_" + str(t) + "s.png", 'PNG')
        print(c)
    
