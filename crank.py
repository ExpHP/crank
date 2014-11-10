#!/usr/bin/env python3
import numpy as np
import scipy.sparse
import scipy.sparse.linalg as spla
import scipy.linalg as la
import itertools
import time
from enum import Enum

from pylibs.debug import memlimit
memlimit(2048)

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
	a = scipy.sparse.dok_matrix((totalSize,totalSize), dtype=complex)

	# value of main diagonal element = -2 * (dx**-2 + dy**-2 + dz**-2 + ...)
	diagonalValue = -2 * sum([dx**-2 for dx in dxs])

	# iterate over matrix rows
	for rowI in range(totalSize):

		a[rowI,rowI] = diagonalValue

		coords = indexToCoords(rowI, strides)

		# Iterate over dimensions
		for dimI,c in enumerate(coords):

			# Find index corresponding to changing this coordinate by 1 in either direction
			targetCoords1 = list(coords);  targetCoords1[dimI] -= 1;
			targetCoords2 = list(coords);  targetCoords2[dimI] += 1;

			for targetCoords in (targetCoords1, targetCoords2):
				# Wrap if periodic
				if bctype is BoundaryType.PERIODIC:
					targetCoords[dimI] %= dims[dimI]

				if (0 <= targetCoords[dimI] < dims[dimI]):
					a[rowI, coordsToIndex(targetCoords, strides)] = dxs[dimI]**-2

	return a.tocsc()

def getHamiltonian(potgrid, bctype, *, lengths=None, mass=1., hbar=1):
	dims = potgrid.shape
	potvec = potgrid.flatten()

	if lengths is None:   # assume all lengths are 1
		lengths = [1. for d in dims]

	if bctype is BoundaryType.REFLECTING:
		dxs = [lengths[i] / (dims[i]-1) for i in range(len(dims))]
	elif bctype is BoundaryType.PERIODIC:
		dxs = [lengths[i] / dims[i] for i in range(len(dims))]
	else: assert False

	return -(hbar*hbar)/(2*mass) * makeLaplacian(dims,dxs,bctype) + scipy.sparse.spdiags(potvec,0,len(potvec),len(potvec))


class getCrankNicolEvo:

	def __init__(self, potgrid, bctype, dt, *, lengths=None, mass=1., hbar=1.):
		strides = getStrides(potgrid.shape)

		assert isinstance(bctype, BoundaryType)
		assert len(potgrid.flatten()) == strides[-1]

		ham = getHamiltonian(potgrid, bctype, lengths=lengths, mass=mass, hbar=hbar)

		self.matL = scipy.sparse.eye(strides[-1]) - (dt/(2.j*hbar)) * ham
		self.matR = scipy.sparse.eye(strides[-1]) + (dt/(2.j*hbar)) * ham

		self.matL = scipy.sparse.csc_matrix(self.matL)
		self.matR = scipy.sparse.csc_matrix(self.matR)

		rsolver = spla.factorized(self.matR)

		self._evofunc = lambda vec: self.matL.dot(rsolver(vec))
		self.dt      = dt
		self.hbar    = hbar
		self.mass    = mass
		self.lengths = lengths
		self.potgrid = potgrid
		self.dims    = potgrid.shape
		self.bctype  = bctype

	def __call__(self, vec):
		return self._evofunc(vec)

def oneDeeGroundState(n):
	# Note this here!  The boundary conditions are left out of the vector
	x = np.linspace(0., np.pi, n+2)[1:-1]
	vec = np.sin(x).astype(complex)
	prefactor = sum(np.abs(vec)**2.) ** (-0.5)
	return prefactor * vec


def twoDeeGroundState(n,m):
	# Note this here!  The boundary conditions are left out of the vector
	y = np.linspace(0., np.pi, n+2)[1:-1]
	x = np.linspace(0., np.pi, m+2)[1:-1]
	vecy = np.sin(y).astype(complex)
	vecx = np.sin(x).astype(complex)
	vec = np.outer(vecy,vecx).flatten()
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

def donutFunc(r, centerRadius, halfWidth):
	return max(0, 1 - (r - centerRadius)**2 / halfWidth**2)
