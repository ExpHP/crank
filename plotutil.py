
from colorsys import hls_to_rgb
import numpy as np
import math

class WavePlotter:
	def plot(self, ax, psigrid, evo):
		raise NotImplementedError
	def updateAnim(self, psigrid, evo):
		raise NotImplementedError

#---------------

def makeProbPlot(ax, psigrid, evo):
	prob = np.power(np.absolute(psigrid),2)
	return ax.imshow(prob, interpolation='nearest')

def updateProbPlot(img, psigrid, evo):
	prob = np.power(np.absolute(psigrid),2)
	img.set_array(prob)
	img.set_clim(0., prob.max()**0.90)

#---------------
	
def makePhasePlot(ax, psigrid, evo):
	return ax.imshow(getPhaseRGB(psigrid), interpolation='nearest')
	
def updatePhasePlot(img, psigrid, evo):
	img.set_array(getPhaseRGB(psigrid))

def getPhaseRGB(z):
	h = (np.angle(z) + math.pi)  / (2 * math.pi) + 0.5

	l = 1.0 - 1.0/(1.0 + np.abs(z)**0.3)
	l = l**2.
	l /= l.max()
	l *= 0.60

	s = 0.8

	c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
	c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
	c = c.transpose(1,2,0)
	return c

#----------------


