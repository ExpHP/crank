#!/usr/bin/env python3
import numpy as np
import scipy.sparse
import scipy.sparse.linalg as spla
import scipy.linalg as la
import itertools
import time
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


def getHamiltonian(potgrid, bctype, *, lengths=None, mass=1., hbar=1):
	dims = potgrid.shape
	potvec = potgrid.flatten()

	if lengths is None:   # assume all lengths are 1
		lengths = [1. for d in dims]

	# XXX (dims[i]-1) will not apply for PERIODIC --- or will it
	dxs = [lengths[i] / (dims[i]-1) for i in range(len(dims))]
	return -(hbar*hbar)/(2*m) * makeLaplacian(dims,dxs,bctype) + np.diag(potvec)


def getCrankNicolEvo(potgrid, bctype, dt, *, lengths=None, mass=1., hbar=1.):

	strides = getStrides(potgrid.shape)

	assert isinstance(bctype, BoundaryType)
	assert len(potgrid.flatten()) == strides[-1]

	ham = getHamiltonian(potgrid, bctype, lengths=lengths, mass=mass, hbar=hbar)

	matL = np.eye(strides[-1]) - (dt/(2.j*hbar)) * ham
	matR = np.eye(strides[-1]) + (dt/(2.j*hbar)) * ham

	matL = scipy.sparse.csc_matrix(matL)
	matR = scipy.sparse.csc_matrix(matR)

	rsolver = spla.factorized(matR)

	return lambda vec: matL.dot(rsolver(vec))



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
	return normalize(vec)


def norm(vec):
	return sum(np.abs(vec)**2.) ** (0.5)

def normalize(vec):
	return vec / norm(vec)


def parallelSort(a, b):
	a, b = zip(*sorted(zip(a, b)))


def getEigens(potgrid, bctype=BoundaryType.REFLECTING, count=8, guess=0.):
	ham = getHamiltonian(potgrid, bctype=bctype)
	ham = scipy.sparse.csc_matrix(ham)

	w,v = spla.eigs(ham,count,sigma=guess)

	evals = [e for e in w]
	epsis = [e for e in v.transpose()]

	parallelSort(evals,epsis)

	return epsis,evals

n = 20
m = 20

L = 1.
mass = 1.
hbar = 1.
dt = 0.001

bctype = BoundaryType.REFLECTING

vec1 = oneDeeGroundState(n).astype(complex)
vec2 = twoDeeGroundState(n,m).astype(complex)

potgrid1 = np.zeros((n,))
evo1 = getCrankNicolEvo(potgrid1, bctype, dt)

potgrid2 = np.zeros((n,m))
evo2 = getCrankNicolEvo(potgrid2, bctype, dt)

vec3 = getEigens(potgrid2, bctype, count=8)[0][3]
#vec3+= getEigens(potgrid2, bctype, count=8)[0][4]
vec3 = normalize(vec3)


for i in range(50):
#	print(np.abs(vec1)[0:3])
	vec1 = evo1(vec1)

for i in range(1000):
#	print(np.abs(vec3)[0:3])
	vec3 = evo2(vec3)

