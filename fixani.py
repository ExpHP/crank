import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.cbook import iterable
import itertools
# copied directly from the proposed fix
def monkey_patch_init(self, fig, func, frames=None, init_func=None, fargs=None,
             save_count=None, **kwargs):
    if fargs:
        self._args = fargs
    else:
        self._args = ()
    self._func = func

    self.save_count = save_count

    if frames is None:
        self._iter_gen = itertools.count
    elif callable(frames):
        self._iter_gen = frames
    elif iterable(frames):
        self._iter_gen = lambda: iter(frames)
        if hasattr(frames, '__len__'):
            self.save_count = len(frames)
    else:
        self._iter_gen = lambda: xrange(frames).__iter__()
        self.save_count = frames

    if self.save_count is None:
        self.save_count = 100

    self._init_func = init_func
    self._save_seq = []
    animation.TimedAnimation.__init__(self, fig, **kwargs)
    self._save_seq = []

animation.FuncAnimation.__init__ = monkey_patch_init
