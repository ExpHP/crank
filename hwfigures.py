import itertools
import random
import matplotlib.pyplot as plt
import numpy as np
import math
import cmath

from crank import *
from util import *
import animate

#  Important figures
#    - Show normalization is preserved
#
#  1. A stationary state of the empty box (REFLECTING)
#  2. A stationary state of the empty box (PERIODIC)
#    - Actually, I don't think those exist.  Try a plane wave instead.
#  3. Non-stationary states in empty box (REFLECTING)
#    - Simple sums of stationary states
#    - Donut?
#    - A sine on one axis?
#  4. (REFLECTING) Time evolution of an empty-box eigenstate with a potential box
#  5. (REFLECTING) A stationary state for the same potential box (as found using an eigensolver)
#  6. (PERIODIC) A regular plane wave with a potential box
#  7. (VARIOUS) Try different numbers of potential boxes
#  8. (PERIODIC) Diffraction of a wave packet entering a small gap  -->  |========  ========|


n = 50
m = 50

L = 1.
mass = 1.
hbar = 1.
dt = 0.0001
#dt = 0.1


def twoDeePlaneWave(shape, xorder, yorder):
	n,m = shape
#	y = np.linspace(0., np.pi, n+2)[1:-1]
#	x = np.linspace(0., np.pi, m+2)[1:-1]
	y = np.linspace(0., 1., n)
	x = np.linspace(0., 1., m)
	vecy = np.exp(2j * np.pi * yorder * y).astype(complex)
	vecx = np.exp(2j * np.pi * xorder * x).astype(complex)
	vec = np.outer(vecy,vecx).flatten()
	return normalize(vec)

#fig = plt.figure()
#ax = fig.add_subplot(111)
#
#boxes = makeRandomBoxes(6, 0.2)
#
#plt.imshow(potgrid)
#plt.show()

bctype = BoundaryType.REFLECTING

vec1 = oneDeeGroundState(n).astype(complex)
vec2 = twoDeeGroundState(n,m).astype(complex)

potgrid1 = np.zeros((n,))
evo1 = getCrankNicolEvo(potgrid1, bctype, dt)

potgrid2 = np.zeros((n,m))
evo2 = getCrankNicolEvo(potgrid2, bctype, dt)

vec3 = getEigens(potgrid2, bctype, count=8)[0][3]
vec3+= getEigens(potgrid2, bctype, count=8)[0][4]
vec3 = normalize(vec3)


for i in range(n):
	for j in range(m):
			ry = float(j)/n - 0.5
			rx = float(i)/m - 0.5
			r = (rx*rx + ry*ry) ** 0.5
#			vec3[i*n+j] = donutFunc(r, 0.2, 0.1)
			vec3[i*n+j] = donutFunc(r, 0, 0.2) * cmath.exp(3j*math.pi*r)
#			vec3[i*n+j] = donutFunc(r, 0, 0.2) * cmath.exp(3j*math.pi*rx)
vec3 = normalize(vec3)
print(vec3.dtype)

evo2 = getCrankNicolEvo(potgrid2, BoundaryType.PERIODIC, dt)
vec3 = twoDeePlaneWave((n,m), 3, 0)

#vec3 = np.zeros(n*m)
#vec3[len(vec3)/2+35]=1.

for i in range(50):
#	print(np.abs(vec1)[0:3])
	vec1 = evo1(vec1)

#for i in range(1000):
#	print(np.abs(vec3)[0:3])
#	vec3 = evo2(vec3)

import cProfile

import animate
#animate.animateTo("cool.mp4", vec3, evo2, 100, 10)
#animate.animateTo("cool3.gif", vec3, evo2, 50, 10)
#animate.animateTo('ballp.mp4', vec3, evo2, 100, 5)
animate.animateTo('plane.mp4', vec3, evo2, 100, 5)
#cProfile.run("animate.animateTo('ball.mp4', vec3, evo2, 1000, 5)", sort='tottime')
