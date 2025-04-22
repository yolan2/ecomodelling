# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:17:15 2019

@author: jrhillae
"""
import matplotlib.pyplot as plt
import numpy as np
import random

x=[ i for i in range(10)]
y=[ i/2 for i in range(10)]
a=np.random.rand(5,5)

fig= plt.figure(figsize=(15,5))
fig1=fig.add_subplot(121)
fig2=fig.add_subplot(122)

fig1.plot(x,y)
fig1.set_title('Example of graph')
fig1.set_xlabel('xlabel')
fig1.set_ylabel('ylabel')

fig2.imshow(a, animated=False, cmap='Greens', interpolation='none', origin="upper")

fig.savefig('test.png', dpi=200)