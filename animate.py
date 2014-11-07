import numpy as np
from colorsys import hls_to_rgb
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import fixani
from plotutil import *

def frameDataGen(model, frameCount, frameStep=100):
	yield model.getTime(), model.getPsiMap()

	for frameI in range(frameCount-1):
		model.step(frameStep)
		yield model.getTime(), model.getPsiMap()

ANI_FIG_OFFSET = 5000
NEXT_ANI_FIG = ANI_FIG_OFFSET

def animate(model, frameCount, frameStep=100):
	global NEXT_ANI_FIG
	fig = plt.figure(NEXT_ANI_FIG)
	NEXT_ANI_FIG += 1

	psi = model.getPsiMap()
	prob = np.power(np.absolute(ps),2)

	ax1 = fig.add_subplot(121, aspect='equal', autoscale_on=False,
	                      xlim=(0,psi.shape[1]), ylim=(0,psi.shape[0]+1))
	ax2 = fig.add_subplot(122, aspect='equal', autoscale_on=False,
	                      xlim=(0,psi.shape[1]), ylim=(0,psi.shape[0]+1))

	img1 = ax1.imshow(prob)
	img2 = ax2.imshow(colorize(psi))
	time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)

	def update(args):
		t, psi = args
		prob = np.power(np.absolute(ps),2)
		img1.set_array(prob)
		img2.set_array(colorize(ps))
		print int(t/dt)
		time_text.set_text('t = {} fs'.format(t))
		return [img1, img2, time_text]

	def init():
		print 'initializing'
#		return update(0., model.getPsiMap())
		wavemap = model.getPsiMap()
		probmap = np.power(np.absolute(wavemap),2)
		img1.set_array(probmap)
		img2.set_array(colorize(wavemap))
		time_text.set_text('')
		return [img1, img2, time_text]

	print 'make datagen'
	datagen = frameDataGen(model, frameCount, frameStep)
	print 'done datagen'
	anim = animation.FuncAnimation(fig, update,
	           init_func=init, frames=datagen, interval=5,
	           save_count=frameCount, blit=True) # nframes-1
	anim.save('test.mp4', fps=60, extra_args=['-vcodec', 'libx264'])#, '-vb', '10000K'])
	#plt.show()
