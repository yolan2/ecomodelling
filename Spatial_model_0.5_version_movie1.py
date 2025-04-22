# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 08:39:37 2014

@author: Jboeye
"""

import random as rnd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Individual:
    """Class that regulates individuals and their properties"""
    def __init__(self,
                 x,
                 y,
                 resources):
        """Initialization"""
        self.x = x
        self.y = y
        self.angle = rnd.uniform(0, 2 * math.pi)
        self.resources = resources
        self.age = 0
        self.reproductive_age = rnd.randint(10, 15)

    def move(self, max_x, max_y):
        """Calculates movement"""
        speed = 1
        diversion = math.pi / 3.0
        self.resources -= 1
        self.angle += rnd.uniform(-diversion, diversion)
        dx = speed * math.cos(self.angle)
        dy = speed * math.sin(self.angle)
        self.x = (self.x + dx) % max_x
        self.y = (self.y + dy) % max_y


class Metapopulation:
    """Contains the whole population, regulates daily affairs"""
    def __init__(self,
                 max_x,
                 max_y):
        """Initialization"""
        self.max_x = max_x
        self.max_y = max_y
        initial_resources = 70
        self.environment = np.zeros((self.max_x, self.max_y)) + initial_resources
        self.population = []
        self.initialize_pop()

        self.x = []
        self.y1 = []
        self.y2 = []

    def initialize_pop(self):
        """Initialize individuals"""
        startpop = 100
        start_resources = 10
        for n in range(startpop):
            x = rnd.uniform(0, self.max_x)
            y = rnd.uniform(0, self.max_y)
            self.population.append(Individual(x, y,
                                              start_resources))

    def a_day_in_the_life(self, timer):
        """Replenish patches and draw visual"""
        dist_pop = np.zeros((self.max_x, self.max_y))
        rnd.shuffle(self.population)
        cost_of_offspring = 10
        # shuffle population so that individuals in the beginning of the list
        # don't get an advantage
        oldpop = self.population[:]
        del self.population[:]
        for indiv in oldpop:
            if indiv.age >= indiv.reproductive_age:
                n_offspring = int(indiv.resources) // cost_of_offspring
                for n in range(n_offspring):
                    self.population.append(Individual(indiv.x,
                                                      indiv.y,
                                                      cost_of_offspring))
                    dist_pop[int(indiv.x), int(indiv.y)] += 1

            else:
                if indiv.resources >= 0:
                    indiv.move(self.max_x, self.max_y)
                    dist_pop[int(indiv.x), int(indiv.y)] += 1
                    if self.environment[int(indiv.x), int(indiv.y)] > 0:
                        if self.environment[int(indiv.x), int(indiv.y)] > 5:
                            self.environment[int(indiv.x), int(indiv.y)] -= 5
                            indiv.resources += 5
                        else:
                            indiv.resources += self.environment[int(indiv.x), int(indiv.y)]
                            self.environment[int(indiv.x), int(indiv.y)] = 0
                    indiv.age += 1
                    self.population.append(indiv)

        self.environment += 2  # replenish resources in patches
        np.clip(self.environment, 0, 100, out=self.environment)
        # amount of resources has to stay between 0 and 100
        self.x.append(timer)
        print(len(self.population))
        self.y1.append(len(self.population))
        self.y2.append(np.mean(self.environment))
        # saving frames of movie
        ima = ax1.imshow(dist_pop, animated=True, cmap='Greens', interpolation='none', origin="upper")
        imb = ax2.imshow(self.environment, animated=True, vmax=100, cmap='YlOrBr', interpolation='none', origin="upper")
        imc = ax3.plot(self.x, self.y1, 'b', animated=True)
        imd = ax4.plot(self.x, self.y2, 'r', animated=True)
        ims.append([ima, imb, imc[0], imd[0]])


meta = Metapopulation(40, 40)
# defining number of figures in movie
ims = []
fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7, 7))

ax1.set_title('Distribution of population')
ax2.set_title('Distribution of resource')
ax3.set_title('Population size')
ax4.set_title('Amount of resources')

for timer in range(100):
    meta.a_day_in_the_life(timer)


# creating and saving movie
ani1 = animation.ArtistAnimation(fig1, ims, interval=250, blit=False, repeat_delay=1000)
ani1.save('mg.gif',  writer='pillow', dpi=200)

print("GIF created")
