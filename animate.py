import numpy as np
from colorsys import hls_to_rgb
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import time
#import fixani
from plotutil import *

def frameDataGen(psi, evo, frameCount, frameStep=1):
	yield 0, psi

	print(psi)
	print(psi.shape)

	for frameI in range(1,frameCount):
		for _ in range(frameStep):
			psi = evo(psi)
		yield frameI*frameStep, psi

ANI_FIG_OFFSET = 5000
NEXT_ANI_FIG = ANI_FIG_OFFSET

# evo is a function that evolves psi by dt
def animateTo(outfilename, psi, evo, frameCount, frameStep=1):
	global NEXT_ANI_FIG
	fig = plt.figure(NEXT_ANI_FIG)
	NEXT_ANI_FIG += 1

	prob = np.power(np.absolute(psi),2)

	ax1 = fig.add_subplot(121, aspect='equal', autoscale_on=False,
	                      xlim=(0,evo.dims[1]), ylim=(0,evo.dims[0]+1))
	ax2 = fig.add_subplot(122, aspect='equal', autoscale_on=False,
	                      xlim=(0,evo.dims[1]), ylim=(0,evo.dims[0]+1))

	img1 = ax1.imshow(prob.reshape(evo.dims))
	img2 = ax2.imshow(colorize(psi.reshape(evo.dims)))
	time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)

	def update(args):
		frame, psi = args
		print('update: frame {}'.format(frame))

		psi = psi.reshape(evo.dims)
		prob = np.power(np.absolute(psi),2)

		img1.set_array(prob)
		img2.set_array(colorize(psi))
		time_text.set_text('t = {}'.format(frame))
		return [img1, img2, time_text]

	def init():
		print('initializing')
		return update((0, psi.reshape(evo.dims)))

	print('make datagen')
	datagen = frameDataGen(psi, evo, frameCount, frameStep)
	print('done datagen')
	anim = animation.FuncAnimation(fig, update,
	           init_func=init, frames=datagen, interval=5,
	           save_count=frameCount, blit=True) # nframes-1
	anim.save(outfilename, fps=60, extra_args=['-vcodec', 'libx264'])#, '-vb', '10000K'])
#	anim.save(outfilename, fps=60, extra_args=['-vcodec', 'libx264', '-vb', '10000K'])
#	anim.save(outfilename, writer='imagemagick', fps=60)
#	plt.show()
