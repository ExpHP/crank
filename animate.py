import numpy as np
from colorsys import hls_to_rgb
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import time
#import fixani
from plotutil import *

def frameDataGen(psi, evo, frameCount, frameStep=1):
	yield 0, psi, evo

	for frameI in range(1,frameCount):
		for _ in range(frameStep):
			psi = evo(psi)
		yield frameI*frameStep, psi, evo

ANI_FIG_OFFSET = 5000
NEXT_ANI_FIG = ANI_FIG_OFFSET

# evo is a function that evolves psi by dt
def animateTo(outfilename, psi, evo, frameCount, frameStep=1):
	global NEXT_ANI_FIG
	fig = plt.figure(NEXT_ANI_FIG)
	NEXT_ANI_FIG += 1

	prob = np.power(np.absolute(psi),2)

	ax1 = fig.add_subplot(121, aspect='equal')
	ax2 = fig.add_subplot(122, aspect='equal')

	psigrid = psi.reshape(evo.dims)
	img1 = makeProbPlot(ax1, psigrid, evo)
	img2 = makePhasePlot(ax2, psigrid, evo)
	time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)

	fig.subplots_adjust(left=0,bottom=0,top=1,right=1,wspace=None,hspace=None)

	def update(args):
		frame, psi, evo = args
		print('update: frame {}'.format(frame))

		psigrid = psi.reshape(evo.dims)
		updateProbPlot(img1, psigrid, evo)
		updatePhasePlot(img2, psigrid, evo)

#		time_text.set_text('t = {}'.format(frame))
		return [img1, img2, time_text]

	def init():
		print('initializing')

	datagen = frameDataGen(psi, evo, frameCount, frameStep)
	anim = animation.FuncAnimation(fig, update,
	           init_func=init, frames=datagen, interval=5,
	           save_count=frameCount, blit=True) # nframes-1
#	anim.save(outfilename, fps=60, extra_args=['-vcodec', 'libx264'])#, '-vb', '10000K'])
#	anim.save(outfilename, fps=60, extra_args=['-vcodec', 'libx264', '-vb', '10000K'])
#	anim.save(outfilename, writer='imagemagick', extra_args=['-size', '{}x{}'.format(*outDims)], fps=30)
	anim.save(outfilename, writer='imagemagick', fps=30)
#	plt.show()
