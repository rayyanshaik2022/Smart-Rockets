import os
import random
import time
import math
from math import atan2, cos, sin, tan

import numpy as np
import pygame
import pygame.gfxdraw
from pygame.math import Vector2 as Vector

pygame.init()
random.seed(time.time())

WIDTH = 800
HEIGHT = 800
FPS = 240

BACKGROUND = (0, 0, 0)

''' Setup 1 '''
LIFESPAN = 550
MUTATION = 0.001 # 0.001
MAXFORCE = 0.2
COUNT = 0
TARGET = Vector(WIDTH/2, 50)
SPAWNPOINT = WIDTH/2, HEIGHT

RECT_OBSTACLES = [
    [200,350,400,50],
    [0,600,150,30],
    [650,600,150,30],
    [450,0,20,350]
]


def random_vector(magnitude=1):

    # Generates a random direction & speed
    '''
    x = random.random()
    if (random.random() > 0.5):
        x *= -1

    y = random.random()
    if (random.random() > 0.5):
        y *= -1

    return Vector(x=x, y=y)
    '''

    # Generates a vector with constant speeds but random direction
    phi = 2 * math.pi * random.random()
    vx = magnitude * cos(phi)
    vy = magnitude * sin(phi)

    return Vector(x=vx, y=vy)


class Population():

    def __init__(self):
        self.rockets = []
        self.popsize = 500 # 500
        self.generation = 1
        self.maxscore = 0
        self.avgscore = 0
        self.matingpool = []
        self.best_rocket = None

        self.rockets = [Rocket(None) for i in range(self.popsize)]
    
    def evaluate(self):

        maxfit = 0
        for rocket in self.rockets:
            rocket.calcFitness()
            if rocket.fitness > maxfit:
                maxfit = rocket.fitness
                self.best_rocket = rocket
        
        self.maxscore = '%.3f' % maxfit
        self.avgscore = '%.3f' % (sum(r.fitness for r in self.rockets) / len(self.rockets))
        print(f"Greatest Fitness: {self.maxscore}\nAverage Fitness: {self.avgscore}\n")
        # Normalize values to be between 0-1
        for rocket in self.rockets:
            rocket.fitness /= maxfit

        self.matingpool = []
        for rocket in self.rockets:
            n = rocket.fitness * 100
            for j in range(int(n)):
                self.matingpool.append(rocket)
    
    def selection(self):
        
        newRockets = []

        for i in range(len(self.rockets)):
            # Get the DNA of 2 random parents
            parentA = random.choice(self.matingpool).dna
            parentB = random.choice(self.matingpool).dna

            # DNA object
            child = parentA.crossover(parentB)
            child.mutation()
            newRockets.append(Rocket(child))
        
        self.rockets = newRockets
        self.best_rocket.color = (69, 205, 247)
        self.best_rocket.reset()
        self.rockets.append(self.best_rocket)

    
    def run(self, screen):
        for i, rocket in enumerate(self.rockets):
            rocket.update()
            rocket.show(screen)

class DNA():

    def __init__(self, gene_o=False):
        self.genes = []
        
        if gene_o:
            self.genes = gene_o
        else:
            for i in range(LIFESPAN):
                self.genes.append(random_vector(MAXFORCE))
    
    def crossover(self, partner):

        newgenes = []
        mid = random.randint(0,len(self.genes))
        for i in range(len(self.genes)):
            # If i > mid, take genes from original dna
            if i > mid:
                newgenes.append( self.genes[i] )
            # Else if < mid, take genes from partner
            else:
                newgenes.append( partner.genes[i] )
        return DNA(newgenes)
    
    def mutation(self):

        for i in range(len(self.genes)):
            if random.random() < MUTATION:
                self.genes[i] = random_vector(MAXFORCE)

class Rocket():

    def __init__(self, dna=None):

        self.pos = Vector(SPAWNPOINT)
        self.vel = Vector()
        self.acc = Vector()
        if dna == None:
            self.dna = DNA()
        else:
            self.dna = dna
        self.fitness = 0
        self.completed = False
        self.crashed = False

        self.count = 0
        self.color = (255,255,255)

    def reset(self):
        self.pos = Vector(SPAWNPOINT)
        self.vel = Vector()
        self.acc = Vector()
        self.fitness = 0
        self.completed, self.crashed = False, False
        self.count = 0

    def applyForce(self, force):
        self.acc += force
    
    # TODO: Redefine fitness calculation
    def calcFitness(self):

        d = math.sqrt( ((self.pos.x-TARGET.x)**2)+((self.pos.y-TARGET.y)**2) ) + 1
        self.fitness = 1/d
        if self.completed:
            # TODO : find a better completion goal
            self.fitness *= 1.01
        if self.crashed:

            # TODO : find a better crash avoidance value
            self.fitness *= 0.1

    def update(self):

        d = math.sqrt( ((self.pos.x-TARGET.x)**2)+((self.pos.y-TARGET.y)**2) )
        if (d < 10):
            self.completed = True
            self.pos.x, self.pos.y = TARGET.x, TARGET.y
        
        # If rocket has hit a rectangle obstacle
        for rect in RECT_OBSTACLES:
            
            if (self.pos.x > rect[0] and self.pos.x < rect[0] + rect[2]) \
                and (self.pos.y > rect[1] and self.pos.y < rect[1] + rect[3]):

                self.crashed = True
        
        # If rocket is out of bounds
        #if (self.pos.x < 0 or self.pos.x > WIDTH) or (self.pos.y < 0 or self.pos.y > HEIGHT):

            #self.crashed = True

        if COUNT < LIFESPAN:
            self.applyForce(self.dna.genes[COUNT])

        if not self.completed and not self.crashed:
            self.vel += self.acc
            self.pos += self.vel
            self.acc *= 0
            self.count += 1

    def show(self, screen):
        
        width = int(5/2)
        length = int(25/2)

        angle = atan2(self.vel.y, self.vel.x)  # Gets angle in radians

        # General points for a rectangle around the center point
        points = [
            [self.pos.x - length, self.pos.y - width],
            [self.pos.x + length, self.pos.y - width],
            [self.pos.x + length, self.pos.y + width],
            [self.pos.x - length, self.pos.y + width],
        ]

        for i, point in enumerate(points):
            # Rotate points around center (x,y)
            points[i] = [int((point[0] - self.pos.x)*cos(angle) - (point[1] - self.pos.y)*sin(angle))+self.pos.x,
                         int((point[0] - self.pos.x)*sin(angle)+(point[1] - self.pos.y)*cos(angle)) + self.pos.y]

        pygame.gfxdraw.aapolygon(screen, points, self.color)
        pygame.draw.polygon(screen, self.color, points)
        pygame.gfxdraw.polygon(screen, points, (200,200,200))


screen = pygame.display.set_mode([WIDTH, HEIGHT])
clock = pygame.time.Clock()
pygame.display.set_caption("Smart Rockets")

''' Setup 2 '''

rocket = Rocket()
population = Population()

running = True
while running:

    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BACKGROUND)
    
    # Draw target
    pygame.gfxdraw.aacircle(screen, int(TARGET.x), int(TARGET.y), 16, (69, 247, 125))
    population.run(screen)

    # Draw rectangle obstacles
    for rectangle in RECT_OBSTACLES:
        pygame.gfxdraw.rectangle(screen, rectangle, (247, 69, 69))

    pygame.display.set_caption(f"Smart Rockets | Generation {population.generation} | Count {COUNT} | Max Fit {population.maxscore} | Avg Fit {population.avgscore}")
    COUNT += 1
    

    if COUNT >= LIFESPAN:
        population.evaluate()
        population.selection() 
        population.generation += 1
        #population = Population()
        COUNT = 0


    

    pygame.display.update()

pygame.quit()
