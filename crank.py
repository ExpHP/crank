#!/usr/bin/env python3
import numpy as np
import scipy.linalg as la
import itertools
from enum import Enum

# TODO: Model
#   model.getPsi      - psi in vector form
#   model.getPsiMap   - psi in 2D form
#   model.step(nstep) - advance by nstep steps
#   model.getTime
#   model.getMapShape


# evo.
# evo.dimensions
# evo.step()

# 

class BoundaryType(Enum):
	REFLECTING = 1
	PERIODIC   = 2



#---------
# Converting between a flattened matrix index and multidimensional coordinates,
#  given the list of strides for motion along each axis.
#
# strides lists the strides for each axis, in FORTRAN order (minor axis first).
# The last element is the total length of the list.
# (e.g. for a matrix of shape 11 x 7 x 5 x 3, strides = [1, 3, 15, 107, 1155])

def getStrides(dims):
	strides = [1] + list(reversed(dims))
	for i in range(1,len(strides)):
		strides[i] *= strides[i-1]
	return strides

def indexToCoords(idx, strides):
	return [(idx % strides[i+1])  //  strides[i]  for i in reversed(range(len(strides)-1))]

def coordsToIndex(xs, strides):
	return sum([strides[-i-2] * xs[i] for i in range(len(xs))])


def makeLaplacian(dims, dxs, bctype):
	assert isinstance(bctype, BoundaryType)

	strides = getStrides(dims)
	totalSize = strides[-1]

	# matrix has 1 row for each point in space
	a = np.zeros((totalSize,totalSize), dtype=complex)

	print("DXs:", dxs)

	# value of main diagonal element = -2 * (dx**-2 + dy**-2 + dz**-2 + ...)
	diagonalValue = -2 * sum([dx**-2 for dx in dxs])

	# iterate over matrix rows
	for rowI in range(totalSize):

		# diagonal elements
		a[rowI,rowI] = diagonalValue #   -2 * len(dims)

		# coordinates of point corresponding to row i:
		coords = indexToCoords(rowI, strides)

		# iterate over matrix rows
		for dimI,c in enumerate(coords):

			# Find index corresponding to changing this coordinate by 1 in either direction
			targetCoords1 = list(coords);  targetCoords1[dimI] -= 1;
			targetCoords2 = list(coords);  targetCoords2[dimI] += 1;

			for targetCoords in (targetCoords1, targetCoords2):
				# wrap if periodic
				if bctype is BoundaryType.PERIODIC:
					targetCoords[dimI] %= dims[dimI]

				if (0 <= targetCoords[dimI] < dims[dimI]):
					a[rowI, coordsToIndex(targetCoords, strides)] = dxs[dimI]**-2

	return a

if False:
	def makeLaplacian(dims, dxs, bctype):
		assert isinstance(bctype, BoundaryType)

		strides = getStrides(dims)
		totalSize = strides[-1]

		# for easily working with the matrix, use two sets of coordinates: e.g. [rowy, rowx, coyx, colx]
		a = np.zeros([*dims]*2, dtype=complex)

		print("DXs:", dxs)

		# value of main diagonal element = -2 * (dx**-2 + dy**-2 + dz**-2 + ...)
		diagonalValue = -2 * sum([dx**-2 for dx in dxs])

		# iterate over matrix "row"s
		for rowI in range(totalSize):

			# diagonal elements
			a[rowI,rowI] = diagonalValue #   -2 * len(dims)

			# coordinates of point corresponding to row i:
			coords = indexToCoords(rowI, strides)

			# iterate over matrix rows
			for dimI,c in enumerate(coords):

				# Find index corresponding to changing this coordinate by 1 in either direction
				targetCoords1 = list(coords);  targetCoords1[dimI] -= 1;
				targetCoords2 = list(coords);  targetCoords2[dimI] += 1;
	
				for targetCoords in (targetCoords1, targetCoords2):
					# wrap if periodic
					if bctype is BoundaryType.PERIODIC:
						targetCoords[dimI] %= dims[dimI]

					if (0 <= targetCoords[dimI] < dims[dimI]):
						a[rowI, coordsToIndex(targetCoords, strides)] = dxs[dimI]**-2

		return a


def getCrankNicolEvo(dims, dt, bctype, pot=None, *, lengths=None, mass=1., hbar=1.):
	strides = getStrides(dims)

	if pot is None:
		pot = np.zeros(strides[-1])

	if lengths is None:
		lengths = [1. for d in dims]

	assert isinstance(bctype, BoundaryType)
	assert len(pot) == strides[-1]

	# more physical parameters beyond the ones supplied
	# XXX (dims[i]-1) will not apply for PERIODIC --- or will it
	dxs = [lengths[i] / (dims[i]-1) for i in range(len(dims))]
	mu = (-2j) * mass / (hbar * dt)


	matL = mu*np.eye(strides[-1]) + makeLaplacian(dims, dxs, bctype)
	matR = mu*np.eye(strides[-1]) - makeLaplacian(dims, dxs, bctype)
	matRInv = la.inv(matR)

	return matRInv.dot(matL)


def oneDeeGroundState(n):
	# Note this here!  The boundary conditions are left out of the vector
	x = np.linspace(0., np.pi, n+2)[1:-1]
	vec = np.sin(x).astype(complex)
	prefactor = sum(np.abs(vec)**2.) ** (-0.5)
	return prefactor * vec

def twoDeeGroundState(n,m):
	# Note this here!  The boundary conditions are left out of the vector
	x = np.linspace(0., np.pi, n+2)[1:-1]
	y = np.linspace(0., np.pi, m+2)[1:-1]
	vecx = np.sin(x).astype(complex)
	vecy = np.sin(y).astype(complex)
	vec = np.outer(vecx,vecy).flatten()
	prefactor = sum(np.abs(vec)**2.) ** (-0.5)
	return prefactor * vec

def flattenTest(n,m):
	# Note this here!  The boundary conditions are left out of the vector
	x = np.linspace(0., np.pi, n+2)[1:-1]
	y = np.linspace(0., np.pi, m+2)[1:-1]
	vecx = np.sin(x).astype(complex)
	vecy = np.sin(y).astype(complex)
	vec = np.outer(vecy,vecx).flatten()
	return normalize(vec)

def normalize(vec):
	prefactor = sum(np.abs(vec)**2.) ** (-0.5)
	return prefactor * vec


n = 20
m = 20
dim = n*m

L = 1.
mass = 1.
hbar = 1.
dt = 0.001

vec1 = oneDeeGroundState(n).astype(complex)
vec2 = twoDeeGroundState(n,15).astype(complex)
mat1 = getCrankNicolEvo((n,), dt, BoundaryType.REFLECTING)
mat2 = getCrankNicolEvo((n,15), dt, BoundaryType.REFLECTING)

for i in range(50):
#	print(sum(np.abs(vec1)**2))
	print(np.abs(vec1)[0:3])
	vec1 = la.solve(mat1, vec1)

#import sys
#sys.exit(0)

for i in range(50):
#	print(sum(np.abs(vec2)**2))
	print(np.abs(vec2)[0:3])
	vec2 = la.solve(mat2, vec2)

