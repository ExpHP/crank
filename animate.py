import numpy as np
from colorsys import hls_to_rgb
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import time
import itertools
from plotutil import *

def frameDataGen(psis, evo, frameCount, frameStep=1):
	yield 0, psis, evo

	for frameI in range(1,frameCount):
		for _ in range(frameStep):
			for i in range(len(psis)):
				psis[i] = evo(psis[i])
		yield frameI*frameStep, psis, evo

ANI_FIG_OFFSET = 5000
NEXT_ANI_FIG = ANI_FIG_OFFSET

def iter2D(it):
	for row in it:
		yield from row

def enumerate2D(it):
	for i,row in enumerate(it):
		for j,elem in enumerate(row):
			yield (i,j,elem)

# Returns 2D array of subplot indices
def getSubplotAxes(fig, rows, cols):
	axes = []
	for i in range(0, rows):
		rowaxes = []
		for j in range(0, cols):
			rowaxes.append(fig.add_subplot(rows, cols, i*cols+j + 1))
		axes.append(rowaxes)
	return axes

# Takes sizes in "inches"
def figsizeForPlotters(rows, cols, plotsize, hspace, vspace):
	w = (hspace+plotsize)*cols - hspace
	h = (vspace+plotsize)*rows - vspace
	return w, h

# evo is a function that evolves psi by dt
def timeEvoAnimation(psis, evo, plotterClasses, frameCount, frameStep=1, dpi=100, interval=50, flip=False):

	global NEXT_ANI_FIG
	fig = plt.figure(NEXT_ANI_FIG)
	NEXT_ANI_FIG += 1

	psis = list(psis)
	plotterClasses = list(plotterClasses)

	if flip:
		axrows = len(plotterClasses)
		axcols = len(psis)
	else:
		axrows = len(psis)
		axcols = len(plotterClasses)

	psiIndices = []
	plotterIndices = []
	for row in range(axrows):
		psiIndices.append([])
		plotterIndices.append([])
		for col in range(axcols):
			if flip: psiIndex = col; plotterIndex = row
			else:    psiIndex = row; plotterIndex = col

			psiIndices[-1].append(psiIndex)
			plotterIndices[-1].append(plotterIndex)

	figsize = figsizeForPlotters(axrows, axcols, 3., 0.2, 0.1)
	fig.set_size_inches(figsize)

	axes = getSubplotAxes(fig, axrows, axcols)
	for ax in iter2D(axes):
		ax.set_aspect('equal')

	plotters = []
	blankgrid = np.ones(evo.dims, dtype=complex) 
	for row in range(axrows):
		rowplotters = []
		for col in range(axcols):
			ax = axes[row][col]
			plotterI = plotterIndices[row][col]
			plotter = plotterClasses[plotterI]() # Instantiate
			plotter.plot(ax, blankgrid, evo)
			rowplotters.append(plotter)
		plotters.append(rowplotters)

#	time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)

#	fig.subplots_adjust(left=0,bottom=0,top=1,right=1,wspace=None,hspace=None)
	fig.tight_layout()

	def update(args):
		frame, psis, evo = args
		psigrids = [psi.reshape(evo.dims) for psi in psis]

		drawables = []
		for row, col in itertools.product(range(axrows), range(axcols)):
			psigrid = psigrids[psiIndices[row][col]]
			plotter = plotters[row][col]
			drawables += plotter.update(psigrid, evo)

		return drawables

	def init():
		pass

	datagen = frameDataGen(psis, evo, frameCount, frameStep)
	anim = animation.FuncAnimation(fig, update,
	           init_func=init, frames=datagen, interval=interval,
	           save_count=frameCount, blit=True) # nframes-1
#	anim.save(outfilename, fps=60, extra_args=['-vcodec', 'libx264'])#, '-vb', '10000K'])
#	anim.save(outfilename, fps=60, extra_args=['-vcodec', 'libx264', '-vb', '10000K'])
#	anim.save(outfilename, writer='imagemagick', fps=30, dpi=dpi)
#	anim.save(outfilename, writer='imagemagick', fps=30)
#	plt.show()
	return anim
