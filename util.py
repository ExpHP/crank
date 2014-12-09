import itertools
import random
import numpy as np

class Box:
	def __init__(self, point1, point2):
		if len(point1) != 2 or len(point2) != 2:
			raise ValueError("Expected two (x,y) pairs")

		self.xmin = min(point1[0], point2[0])
		self.ymin = min(point1[1], point2[1])
		self.xmax = max(point1[0], point2[0])
		self.ymax = max(point1[1], point2[1])

	def containspoint(self, other):
		if len(other) != 2:
			raise ValueError("Expected an (x,y) pair")
		
		x,y = other
		return (self.xmin <= x <= self.xmax) and (self.ymin <= y <= self.ymax)

	def containsbox(self, other):
		if not isinstance(other, Box):
			raise ValueError("Expected a box")

		return (self.xmin <= other.xmin <= other.xmax <= self.xmax and
		        self.ymin <= other.ymin <= other.ymax <= self.ymax)

	__contains__ = containspoint

	def overlaps(self, other):
		if not isinstance(other, Box):
			raise ValueError("Expected a box")
		
		# Two boxes overlap if, on each axis, both min coords are less than both max coords
		xmins = (self.xmin, other.xmin)
		xmaxs = (self.xmax, other.xmax)
		ymins = (self.ymin, other.ymin)
		ymaxs = (self.ymax, other.ymax)
		for xmin,xmax in itertools.product(xmins, xmaxs):
			if xmax < xmin:
				return False

		for ymin,ymax in itertools.product(ymins, ymaxs):
			if ymax < ymin:
				return False

		return True

	def __repr__(self):
		return "Box(({},{}),({},{}))".format(self.xmin,self.ymin, self.xmax, self.ymax)


def makeRandomBoxes(countrange, sizerange, *, xlim=(0.,1.), ylim=(0.,1.), maxFailures = 999):
	assert len(xlim)==2 and xlim[0] < xlim[1]
	assert len(ylim)==2 and ylim[0] < ylim[1]

	# allow specifying a fixed size
	if isinstance(sizerange, (float, int)):
		sizerange = (sizerange, sizerange)

	# allow specifying a fixed count
	if isinstance(countrange, int):
		countrange = (countrange, countrange+1)

	if sizerange[1] > min(xlim[1]-xlim[0], ylim[1]-ylim[0]):
		raise ValueError("Max size in sizerange is larger than the available space!")

	boundsbox = Box((xlim[0], ylim[0]), (xlim[1], ylim[1]))

	failedAttempts = 0
	boxes = []

	count = random.randrange(*countrange)
	while len(boxes) < count:
		w = random.uniform(*sizerange)
		h = random.uniform(*sizerange)

		x1 = random.uniform(xlim[0], xlim[1]-w)
		y1 = random.uniform(ylim[0], ylim[1]-h)

		newbox = Box((x1, y1), (x1+w, y1+h))

		assert boundsbox.containsbox(newbox)

		for oldbox in boxes:
			if newbox.overlaps(oldbox):
				failedAttempts += 1
				break
		else:
			boxes.append(newbox)

		if failedAttempts >= maxFailures:
			raise RuntimeError("maxFailures={} reached when trying to generate boxes".format(maxFailures))

	return boxes


# FIXME: Assumes limits on each axis are 0. to 1., even though some other parts of code
#   allow lengths to be specified.
def potentialFromBoxes(shape, boxes, potentials):
	if len(shape) != 2:
		raise RuntimeError("Only implemented for 2D")

	# allow boxes and potentials to either be iterables or single items
	# (note: if at any point I ever make the Box class iterable, thus breaking this portion
	#    of code entirely, then allow me to say this:  "I told me so!")
	try:    boxes = list(boxes)
	except: boxes = [boxes]
	try:    potentials = list(potentials)
	except: potentials = [potentials]

	if len(boxes) != len(potentials):
		raise RuntimeError("Number of boxes must match number of potentials")

	# Brute force method.  O(n*m*p) (p being number of potentials)
	# This could be made O(log(n) log(m) p) by using bisection, but this code isn't exactly critical.
	n,m = shape
	potgrid = np.zeros(shape)
	for i,j in itertools.product(range(n),range(m)):

		# FIXME: technically, this method needs to be aware of the boundary condition.
		# For REFLECTING, spacing is 1/(N+1), and full space at each end.  |---o---o---o---|
		# For PERIODIC,   spacing is 1/N, with a half-space at each end.   |-o---o---o---o-|
		x,y = (float(j+1)/(m+1), float(i+1)/(n+1)) # suitable for REFLECTING

		for boxidx in range(len(boxes)):
			if (x,y) in boxes[boxidx]:
				potgrid[i,j] = potentials[boxidx]
				break
	return potgrid
