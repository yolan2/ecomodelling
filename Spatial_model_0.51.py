# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 08:39:37 2014
test
@author: Jboeye
"""

import random as rnd
import tkinter as tk
import numpy as np
import math as math
import io
from PIL import Image, ImageSequence

# !! download and install ghostscript 64 bit https://ghostscript.com/releases/gsdnld.html if you get this error !!


class Visual:
    """This class arranges the visual output."""
    def __init__(self, max_x, max_y):
        """Initialize the visual class"""
        self.zoom = 15
        self.max_x = max_x
        self.max_y = max_y
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root,
                                width=self.max_x * self.zoom,
                                height=self.max_y * self.zoom)  # create window
        self.canvas.pack()
        self.canvas.config(background='white')
        self.squares = np.empty((self.max_x, self.max_y), dtype=object)
        self.initialize_squares()

    def create_individual(self, x, y):
        """Create circle for individual"""
        radius = 0.1
        return self.canvas.create_oval((x - radius) * self.zoom,
                                       (y - radius) * self.zoom,
                                       (x + radius) * self.zoom,
                                       (y + radius) * self.zoom,
                                       outline='black',
                                       fill='black')

    def move_drawing(self, drawing, x, y):
        radius = 0.1
        self.canvas.coords(drawing, (x - radius) * self.zoom,
                                    (y - radius) * self.zoom,
                                    (x + radius) * self.zoom,
                                    (y + radius) * self.zoom)

    def color_square(self, resources, x, y):
        """Changes the color of the square"""
        min_res = 1
        max_res = 5
        color = (resources - min_res) / float(max_res - min_res)
        color = max(0, min(1, color))

        green = int(255 * color)
        red = 255 - green
        blue = 0
        rgb = red, green, blue
        hex_code = '#%02x%02x%02x' % rgb
        self.canvas.itemconfigure(self.squares[x, y], fill=str(hex_code))

    def initialize_squares(self):
        """returns a square (drawing object)"""
        for x in range(self.max_x):
            for y in range(self.max_y):
                self.squares[x, y] = self.canvas.create_rectangle(self.zoom * x,
                                                                  self.zoom * y,
                                                                  self.zoom * x + self.zoom,
                                                                  self.zoom * y + self.zoom,
                                                                  outline='black',
                                                                  fill='black')


class Individual:
    """Class that regulates individuals and their properties"""
    def __init__(self,
                 x,
                 y,
                 resources,
                 drawing, speed1, speed2):
        """Initialization"""
        self.speed1 = speed1
        self.speed2 = speed2
        self.x = x
        self.y = y
        self.angle = rnd.uniform(0, 2 * math.pi)
        self.resources = resources
        self.drawing = drawing
        self.age = 0
        self.reproductive_age = rnd.randint(10, 15)
        if self.speed1 == rnd.randint(1, 10):
            self.speed1 = np.random.poisson(lam=self.speed1)
        if self.speed2/2 == rnd.randint(1, 10):
            self.speed2 = np.random.poisson(lam=self.speed2)

    def move(self, max_x, max_y, environment):
        """Calculates movement"""
        min_x_kernel = max(0, int(self.x - 1.1))
        max_x_kernel = min(environment.shape[0] - 1, int(self.x + 1.1))
        min_y_kernel = max(0, int(self.y - 1.1))
        max_y_kernel = min(environment.shape[1] - 1, int(self.y + 1.1))

        sub_grid = environment[min_x_kernel:max_x_kernel + 1, min_y_kernel:max_y_kernel + 1]
        avg_resources = np.mean(sub_grid)

        if rnd.randint(0, 30) == rnd.randint(0, 30):
            speed = np.random.poisson(lam=self.speed2)

        else:
            speed = np.random.poisson(lam=self.speed1)

        self.resources -= (speed / 30)

        diversion = math.pi / 3.0

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
        self.visual = Visual(self.max_x, self.max_y)
        initial_resources = 5
        noise = np.random.rand(self.max_x, self.max_y)

        # Manually smooth the noise to create clustering (3x3 average)
        padded = np.pad(noise, pad_width=1, mode='wrap')
        smoothed = (
                           padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
                           padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
                           padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
                   ) / 9.0

        # Threshold to get 60% resource coverage
        threshold = np.percentile(smoothed, 40)  # 60% will be above this value
        self.environment = np.where(smoothed > threshold, initial_resources, 0.0)
        self.population = []
        self.initialize_pop()
        self.saved_frames = []
        self.avg_speeds1 = []
        self.avg_speeds2 = []


    def initialize_pop(self):
        """Initialize individuals"""
        startpop = 200
        start_resources = 10
        for n in range(startpop):
            x = rnd.uniform(0, self.max_x)
            y = rnd.uniform(0, self.max_y)
            drawing = self.visual.create_individual(x, y)
            speed1 = rnd.randint(1, 10)
            self.population.append(Individual(x, y,
                                              start_resources,
                                              drawing, speed1= speed1, speed2 = 2*speed1))

    def a_day_in_the_life(self):
        """Replenish patches and draw visual"""
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
                    drawing = self.visual.create_individual(indiv.x, indiv.y)
                    self.population.append(Individual(indiv.x,
                                                      indiv.y,
                                                      cost_of_offspring,
                                                      drawing, speed1= indiv.speed1, speed2= indiv.speed2))
                # parents die after reproducing
                self.visual.canvas.delete(indiv.drawing)
            else:
                if indiv.resources >= 0:
                    indiv.move(self.max_x, self.max_y, self.environment)
                    self.visual.move_drawing(indiv.drawing,
                                             indiv.x,
                                             indiv.y)
                    if self.environment[int(indiv.x), int(indiv.y)] > 0:
                        if self.environment[int(indiv.x), int(indiv.y)] > 5:
                            self.environment[int(indiv.x), int(indiv.y)] -= 5
                            indiv.resources += 5
                        else:
                            indiv.resources += self.environment[int(indiv.x), int(indiv.y)]
                            self.environment[int(indiv.x), int(indiv.y)] = 0.1
                    indiv.age += 1
                    self.population.append(indiv)
                else:
                    self.visual.canvas.delete(indiv.drawing)

        for x in range(self.max_x):
            for y in range(self.max_y):
                self.visual.color_square(self.environment[x, y], x, y)
        self.environment[self.environment != 0] += 0.1
        np.clip(self.environment, 0, 100, out=self.environment)
        # amount of resources has to stay between 0 and 100
        print(len(self.population))
        self.visual.root.update()

        # Saving the frames so a GIF can be created afterward
        postscript = self.visual.canvas.postscript(colormode='color')
        image = Image.open(io.BytesIO(postscript.encode('utf-8')))
        self.saved_frames.append(image)

        avg_speed1 = np.mean([indiv.speed1 for indiv in self.population])
        avg_speed2 = np.mean([indiv.speed2 for indiv in self.population])
        self.avg_speeds1.append(avg_speed1)
        self.avg_speeds2.append(avg_speed2)
        print(f"Average speed: {avg_speed1}")
        print(f"Average speed: {avg_speed2}")


meta = Metapopulation(40, 40)
for timer in range(500):
    meta.a_day_in_the_life()

# GIF creation
meta.saved_frames[0].save("output.gif", format='GIF', append_images=meta.saved_frames[1:], save_all=True,
                          duration=100, loop=1)
