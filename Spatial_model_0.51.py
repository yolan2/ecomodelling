# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 08:39:37 2014
test
@author: Jboeye
Updated May 2025: added population size tracking and enhanced plots with distinct colors
"""

import random as rnd
import tkinter as tk
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image


class Visual:
    """This class arranges the visual output."""

    def __init__(self, max_x, max_y):
        self.zoom = 15
        self.max_x = max_x
        self.max_y = max_y
        self.root = tk.Tk()
        self.canvas = tk.Canvas(
            self.root,
            width=self.max_x * self.zoom,
            height=self.max_y * self.zoom
        )
        self.canvas.pack()
        self.canvas.config(background='white')
        self.squares = np.empty((self.max_x, self.max_y), dtype=object)
        self.initialize_squares()

    def create_individual(self, x, y):
        radius = 0.1
        return self.canvas.create_oval(
            (x - radius) * self.zoom,
            (y - radius) * self.zoom,
            (x + radius) * self.zoom,
            (y + radius) * self.zoom,
            outline='black',
            fill='black'
        )

    def move_drawing(self, drawing, x, y):
        radius = 0.1
        self.canvas.coords(
            drawing,
            (x - radius) * self.zoom,
            (y - radius) * self.zoom,
            (x + radius) * self.zoom,
            (y + radius) * self.zoom
        )

    def initialize_squares(self):
        for x in range(self.max_x):
            for y in range(self.max_y):
                self.squares[x, y] = self.canvas.create_rectangle(
                    self.zoom * x,
                    self.zoom * y,
                    self.zoom * x + self.zoom,
                    self.zoom * y + self.zoom,
                    outline='black',
                    fill='black'
                )


class Individual:
    """Class that regulates individuals and their properties"""

    def __init__(self, x, y, resources, drawing, speed1, prob):
        self.x, self.y = x, y
        self.resources = resources
        self.drawing = drawing
        self.speed1 = speed1
        self.prob = prob
        self.angle = rnd.uniform(0, 2 * math.pi)
        self.age = 0
        self.reproductive_age = rnd.randint(10, 15)
        self.teleport_count = 0
        self.movement_count = 0
        self.total_movements = 0

        if rnd.randint(1, 5) == rnd.randint(1, 5) and self.speed1 > 1:
            self.speed1 = rnd.randint(self.speed1 - 1, self.speed1 + 1)
        if rnd.randint(1, 5) == rnd.randint(1, 5) and self.prob > 0.04:
            self.prob = rnd.uniform(self.prob - 0.04, self.prob + 0.04)

    def move(self, max_x, max_y, environment):
        min_x, max_x_k = max(0, int(self.x - 1.1)), min(environment.shape[0] - 1, int(self.x + 1.1))
        min_y, max_y_k = max(0, int(self.y - 1.1)), min(environment.shape[1] - 1, int(self.y + 1.1))
        sub_grid = environment[min_x:max_x_k+1, min_y:max_y_k+1]
        avg_resources = np.mean(sub_grid)

        if rnd.random() < self.prob and avg_resources < 1.5:
            self.x = rnd.uniform(0, max_x)
            self.y = rnd.uniform(0, max_y)
            self.resources -= 1.5
            self.teleport_count += 1
            self.total_movements += 1
        else:
            speed = np.random.poisson(lam=self.speed1)
            self.resources -= 1 + 0.05 * speed
            diversion = math.pi / 3
            self.angle += rnd.uniform(-diversion, diversion)
            dx = speed * math.cos(self.angle)
            dy = speed * math.sin(self.angle)
            self.x = (self.x + dx) % max_x
            self.y = (self.y + dy) % max_y
            self.movement_count += 1
            self.total_movements += 1


class Metapopulation:
    """Contains the whole population, regulates daily affairs"""

    def __init__(self, max_x, max_y):
        self.max_x, self.max_y = max_x, max_y
        self.visual = Visual(max_x, max_y)
        initial_resources = 3
        noise = np.random.rand(max_x, max_y)
        padded = np.pad(noise, pad_width=1, mode='wrap')
        smoothed = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
            padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
            padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        ) / 9.0

        threshold = np.percentile(smoothed, 40)
        self.environment = np.where(smoothed > threshold, initial_resources, 0.0)
        self.population = []
        self.population_sizes = []
        self.initialize_pop()
        self.avg_speeds1 = []
        self.avg_probs2 = []
        self.avg_environment = []

    def initialize_pop(self):
        startpop, start_resources = 200, 5
        for _ in range(startpop):
            x, y = rnd.uniform(0, self.max_x), rnd.uniform(0, self.max_y)
            d = self.visual.create_individual(x, y)
            s = rnd.randint(1, 5)
            p = rnd.random()
            self.population.append(Individual(x, y, start_resources, d, s, p))

    def a_day_in_the_life(self):
        rnd.shuffle(self.population)
        old = self.population[:]
        self.population.clear()

        for indiv in old:
            if indiv.age >= indiv.reproductive_age:
                cost = 10
                n_off = int(indiv.resources) // cost
                for _ in range(n_off):
                    d = self.visual.create_individual(indiv.x, indiv.y)
                    self.population.append(Individual(indiv.x, indiv.y, cost, d, indiv.speed1, indiv.prob))
                self.visual.canvas.delete(indiv.drawing)
            else:
                if indiv.resources >= 0:
                    indiv.move(self.max_x, self.max_y, self.environment)
                    self.visual.move_drawing(indiv.drawing, indiv.x, indiv.y)
                    ix, iy = int(indiv.x), int(indiv.y)
                    if self.environment[ix, iy] > 0:
                        if self.environment[ix, iy] > 5:
                            self.environment[ix, iy] -= 5
                            indiv.resources += 5
                        else:
                            indiv.resources += self.environment[ix, iy]
                            self.environment[ix, iy] = 0.1
                    indiv.age += 1
                    self.population.append(indiv)
                else:
                    self.visual.canvas.delete(indiv.drawing)

        self.environment[self.environment != 0] += 0.1
        np.clip(self.environment, 0, 100, out=self.environment)

        self.avg_speeds1.append(np.mean([i.speed1 for i in self.population]))
        self.avg_probs2.append(np.mean([i.prob for i in self.population]))
        self.avg_environment.append(np.mean(self.environment))
        self.population_sizes.append(len(self.population))


# Simulation parameters
num_simulations = 60
days_per_simulation = 3000

all_avg_speeds = []
all_avg_probs = []
all_avg_environments = []
all_population_sizes = []

for sim in range(num_simulations):
    print(f"Running simulation {sim+1}/{num_simulations}...")
    meta = Metapopulation(40, 40)
    for _ in range(days_per_simulation):
        meta.a_day_in_the_life()
    all_avg_speeds.append(meta.avg_speeds1)
    all_avg_probs.append(meta.avg_probs2)
    all_avg_environments.append(meta.avg_environment)
    all_population_sizes.append(meta.population_sizes)
    rel_tel = np.mean([i.teleport_count/i.total_movements for i in meta.population if i.total_movements>0])
    rel_mov = np.mean([i.movement_count/i.total_movements for i in meta.population if i.total_movements>0])
    print(f"Simulation {sim+1}: Avg rel teleports={rel_tel:.2f}, rel movements={rel_mov:.2f}")

# compute cross-sim averages
all_avg_speeds = np.array(all_avg_speeds)
all_avg_probs = np.array(all_avg_probs)
all_avg_environments = np.array(all_avg_environments)
all_population_sizes = np.array(all_population_sizes)

final_avg_speeds = np.nanmean(all_avg_speeds, axis=0)
final_avg_probs = np.nanmean(all_avg_probs, axis=0)
final_avg_environments = np.nanmean(all_avg_environments, axis=0)
final_avg_populations = np.nanmean(all_population_sizes, axis=0)

# 1: Average Speed
plt.figure()
plt.plot(final_avg_speeds, label='Average Speed', color='blue')
plt.title('Average Speed Over Time')
plt.xlabel('Day')
plt.ylabel('Average Speed')
plt.grid()

# 2: Average Probability of Teleporting
plt.figure()
plt.plot(final_avg_probs, label='Avg Probability Teleporting', color='orange')
plt.title('Average Probability of Teleporting Over Time')
plt.xlabel('Day')
plt.ylabel('Average Probability')
plt.grid()

# 3: Food Availability
plt.figure()
plt.plot(final_avg_environments, label='Food Availability', color='green')
plt.title('Food Availability Over Time')
plt.xlabel('Day')
plt.ylabel('Food Availability')
plt.grid()

# 4: Food vs Speed overlay
plt.figure()
ax1 = plt.gca()
ax2 = ax1.twinx()
line1, = ax1.plot(final_avg_speeds, label='Average Speed', color='blue')
line2, = ax2.plot(final_avg_environments, label='Food Availability', color='green')
ax1.set_xlabel('Day')
ax1.set_ylabel('Average Speed')
ax2.set_ylabel('Food Availability')
plt.legend(handles=[line1, line2], loc='upper right')
plt.title('Food Availability vs Average Speed')
plt.grid()

# 5: Food vs Teleport Probability overlay
plt.figure()
ax3 = plt.gca()
ax4 = ax3.twinx()
line3, = ax3.plot(final_avg_probs, label='Avg Prob Teleporting', color='orange')
line4, = ax4.plot(final_avg_environments, label='Food Availability', color='green')
ax3.set_xlabel('Day')
ax3.set_ylabel('Avg Prob Teleporting')
ax4.set_ylabel('Food Availability')
plt.legend(handles=[line3, line4], loc='upper right')
plt.title('Food Availability vs Teleport Probability')
plt.grid()

# 6: Population vs Speed overlay
plt.figure()
ax5 = plt.gca()
ax6 = ax5.twinx()
line5, = ax5.plot(final_avg_speeds, label='Average Speed', color='blue')
line6, = ax6.plot(final_avg_populations, label='Population Size', color='red')
ax5.set_xlabel('Day')
ax5.set_ylabel('Average Speed')
ax6.set_ylabel('Population Size')
plt.legend(handles=[line5, line6], loc='upper right')
plt.title('Population Size vs Average Speed')
plt.grid()

# 7: Teleport Probability vs Speed overlay
plt.figure()
ax7 = plt.gca()
ax8 = ax7.twinx()
line7, = ax7.plot(final_avg_speeds, label='Average Speed', color='blue')
line8, = ax8.plot(final_avg_probs, label='Avg Prob Teleporting', color='orange')
ax7.set_xlabel('Day')
ax7.set_ylabel('Average Speed')
ax8.set_ylabel('Avg Probability Teleporting')
plt.legend(handles=[line7, line8], loc='upper right')
plt.title('Teleport Probability vs Average Speed')
plt.grid()

plt.tight_layout()
plt.show()
