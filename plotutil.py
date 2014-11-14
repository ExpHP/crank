
from colorsys import hls_to_rgb
import numpy as np
import math

class WavePlotter:
	def plot(self, ax, psigrid, evo):
		''' Produces a plot on ax.  Returns the produced drawable, and records it for later modification through update(). '''
		raise NotImplementedError
	def update(self, psigrid, evo):
		''' Updates the plot from the most recent call to plot() with new data, and returns a list of drawables to blit. '''
		raise NotImplementedError

def clearAxisLabels(ax):
	ax.set_xlabel('')
	ax.set_ylabel('')
	ax.set_xticklabels([])
	ax.set_yticklabels([])

#---------------

class ProbPlotter(WavePlotter):
	def __init__(self):
		self.img = None

	def plot(self, ax, psigrid, evo):
		clearAxisLabels(ax)
		ax.set_title('Probability')
		prob     = np.power(np.absolute(psigrid),2)
		self.img = ax.imshow(prob, interpolation='nearest')
		return self.img

	def update(self, psigrid, evo):
		if self.img is None:
			raise RuntimeError("No existing plot to update")

		prob = np.power(np.absolute(psigrid),2)
		self.img.set_array(prob)
		self.img.set_clim(0., prob.max()**0.97)
		return [self.img]

#---------------

class PhasePlotter(WavePlotter):
	def __init__(self):
		self.img = None

	def plot(self, ax, psigrid, evo):
		clearAxisLabels(ax)
		ax.set_title('Phase')
		self.img = ax.imshow(getPhaseRGB(psigrid), interpolation='nearest')
		return self.img

	def update(self, psigrid, evo):
		if self.img is None:
			raise RuntimeError("No existing plot to update")

		self.img.set_array(getPhaseRGB(psigrid))
		return [self.img]

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


