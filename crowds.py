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
                 numWallsMax = 0,
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
                self.walls.append((c1, c2))
        numWalls = int(numWallsMax * random.random())
        for i in range(numWalls):
            x1 = random.random() * (areaDim[0] - 2 * pedRadius) + \
                 pedRadius
            x2 = random.random() * (areaDim[0] - 2 * pedRadius) + \
                 pedRadius
            y1 = random.random() * (areaDim[1] - 2 * pedRadius) + \
                 pedRadius
            y2 = random.random() * (areaDim[1] - 2 * pedRadius) + \
                 pedRadius
            wall = (np.array([x1, y1]), (np.array([x2, y2])))
            self.walls.append(wall)
    
    def __repr__(self):
        strout = "Number of Pedestrians: " + \
                 str(len(self.pedestrians))
        for p in self.pedestrians:
            strout += "\n" + str(p)
        return strout
    
    def __str__(self):
        return self.__repr__()
    
    def renderScene(self, im, width, height):
        draw = ImageDraw.ImageDraw(im)
        draw.setink("#000000")
        for p in self.pedestrians:
            # Draw the pedestrian's goal
            goalTopLeft = p.goal.pos - np.array([p.radius, p.radius])
            goalBotRight = p.goal.pos - np.array([-p.radius, -p.radius])
            gBounds = (goalTopLeft[0] * width, goalTopLeft[1] * height,
                       goalBotRight[0] * width, goalBotRight[1] * height)
            draw.rectangle(gBounds, fill = p.color)
        
        for p in self.pedestrians:
            # Draw the pedestrian
            eTopLeft = p.pos - np.array([p.radius, p.radius])
            eBotRight = p.pos - np.array([-p.radius, -p.radius])
            eBounds = (eTopLeft[0] * width, eTopLeft[1] * height,
                       eBotRight[0] * width, eBotRight[1] * height)
            draw.ellipse(eBounds, fill = p.color)
            draw.ellipse(eBounds)
        
        for i in range(len(self.pedestrians)):
            p = self.pedestrians[i]
            # Draw the pedestrian's velocity
            xBounds = [p.pos[0] * width,
                       max((p.pos[0] + p.vel[0]) * width, 0)]
            yBounds = [p.pos[1] * height,
                       max((p.pos[1] + p.vel[1]) * height, 0)]
            if not (isinf(np.dot(xBounds, xBounds)) or
                    isinf(np.dot(yBounds, yBounds))):
                aBounds = tuple(zip(xBounds, yBounds))
                draw.line(aBounds, width = 1)
            
            # Draw the force from the goal
            gForce = p.goal.calcForceToPed(p)
            xBounds = [p.pos[0] * width,
                       max((p.pos[0] + gForce[0]) * width, 0)]
            yBounds = [p.pos[1] * height,
                       max((p.pos[1] + gForce[1]) * height, 0)]
            if not (isinf(np.dot(xBounds, xBounds)) or
                    isinf(np.dot(yBounds, yBounds))):
                gfBounds = tuple(zip(xBounds, yBounds))
                draw.line(gfBounds, fill = p.color, width = 4)
            
            # Draw the total force acting on the pedestrian
            otherPeds = self.pedestrians[:i] + self.pedestrians[i + 1:]
            force = p.calcForces(otherPeds, self.walls)
            xBounds = [p.pos[0] * width,
                       max((p.pos[0] + force[0]) * width, 0)]
            yBounds = [p.pos[1] * height,
                       max((p.pos[1] + force[1]) * height, 0)]
            if not isinf(np.dot(force, force)):
                fBounds = tuple(zip(xBounds, yBounds))
                draw.line(fBounds, width = 2)
        
        scaling = [width, height]
        for w in self.walls:
            wt = tuple(tuple(w[i][j] * scaling[j]
                             for j in range(len(w[i])))
                       for i in range(len(w)))
            draw.line(wt, width = 2)
        return self
    
    def timestep(self):
        for i in range(len(self.pedestrians)):
            otherPeds = self.pedestrians[:i] + self.pedestrians[i + 1:]
            self.pedestrians[i].update(otherPeds, self.walls)

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
    frame = 0
    for t in np.linspace(0.0, 10.0, 101):
        im = createImage(imWidth, imHeight)
        c.renderScene(im, imWidth, imHeight)
        c.timestep()
        fname = "Frame_" + format(frame, "03") + ".png"
        frame += 1
        im.save(fname)
        print(fname)
