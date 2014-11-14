import numpy as np
from colorsys import hls_to_rgb
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import time
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
def figsizeForPlotters(psis, plotters, plotsize, hspace, vspace):
	w = (hspace+plotsize)*len(plotters) - hspace
	h = (vspace+plotsize)*len(psis) - vspace
	return w, h

# evo is a function that evolves psi by dt
def timeEvoAnimation(psis, evo, plotterClasses, frameCount, frameStep=1, dpi=100, interval=50):

	global NEXT_ANI_FIG
	fig = plt.figure(NEXT_ANI_FIG)
	NEXT_ANI_FIG += 1

	psis = list(psis)
	plotterClasses = list(plotterClasses)

	figsize = figsizeForPlotters(psis, plotterClasses, 3., 0.2, 0.1)
	fig.set_size_inches(figsize)

	axes = getSubplotAxes(fig, len(psis), len(plotterClasses))
	for ax in iter2D(axes):
		ax.set_aspect('equal')

	plotters = []
	blankgrid = np.ones(evo.dims, dtype=complex) 
	for psiI, ax in enumerate(axes):
		rowplotters = []
		for plotterI, ax in enumerate(row):
			plotter = plotterClasses[plotterI]() # Instantiate
			plotter.plot(ax, blankgrid, evo)
			rowplotters.append(plotter)
		plotters.append(rowplotters)
#		psigrid = psis[psiI].reshape(evo.dims)
#		plotters[plotterI].plot(ax, psigrid, evo)

#	time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)


#	fig.subplots_adjust(left=0,bottom=0,top=1,right=1,wspace=None,hspace=None)

	def update(args):
		frame, psis, evo = args
		psigrids = [psi.reshape(evo.dims) for psi in psis]

		drawables = []
		for rowI, psigrid in enumerate(psigrids):
			for plotter in plotters[rowI]:
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
