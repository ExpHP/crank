import numpy as np
from colorsys import hls_to_rgb
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import time
from plotutil import *

def frameDataGen(psi, evo, frameCount, frameStep=1):
	yield 0, psi, evo

	for frameI in range(1,frameCount):
		for _ in range(frameStep):
			psi = evo(psi)
		yield frameI*frameStep, psi, evo

ANI_FIG_OFFSET = 5000
NEXT_ANI_FIG = ANI_FIG_OFFSET

def getSubplotAxes(fig, rows, cols):
	axes = []
	for i in range(1, rows*cols+1):
		axes.append(fig.add_subplot(rows, cols, i))
	return axes

# Takes sizes in "inches"
def figsizeForPlotters(plotters, height, space):
	width = (space+height)*len(plotters) - space
	return width, height

# evo is a function that evolves psi by dt
def timeEvoAnimation(psi, evo, plotters, frameCount, frameStep=1, dpi=100, interval=50):
	global NEXT_ANI_FIG
	fig = plt.figure(NEXT_ANI_FIG)
	NEXT_ANI_FIG += 1

	figsize = figsizeForPlotters(plotters, 3., 0.2)
	fig.set_size_inches(figsize)

	axes = getSubplotAxes(fig, 1, len(plotters))
	for ax in axes:
		ax.set_aspect('equal')

	psigrid = psi.reshape(evo.dims)
	for i,plotter in enumerate(plotters):
		plotter.plot(axes[i], psigrid, evo)

#	time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)


	fig.subplots_adjust(left=0,bottom=0,top=1,right=1,wspace=None,hspace=None)

	def update(args):
		frame, psi, evo = args
		psigrid = psi.reshape(evo.dims)

		drawables = []
		for plotter in plotters:
			drawables += plotter.update(psigrid, evo)

		return drawables

	def init():
		pass

	datagen = frameDataGen(psi, evo, frameCount, frameStep)
	anim = animation.FuncAnimation(fig, update,
	           init_func=init, frames=datagen, interval=interval,
	           save_count=frameCount, blit=True) # nframes-1
#	anim.save(outfilename, fps=60, extra_args=['-vcodec', 'libx264'])#, '-vb', '10000K'])
#	anim.save(outfilename, fps=60, extra_args=['-vcodec', 'libx264', '-vb', '10000K'])
#	anim.save(outfilename, writer='imagemagick', fps=30, dpi=dpi)
#	anim.save(outfilename, writer='imagemagick', fps=30)
#	plt.show()
	return anim
