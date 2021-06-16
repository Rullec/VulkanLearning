'''
    This script can capture lots of passive IR images, which can be used to do calibration
'''
import matplotlib.pyplot as plt
from log_util import *


class DynaPlotter:
    '''
        Dynamic multiple image plotter based on matplotlib
    '''
    def __init__(self, rows, cols, supress_title="supress_title_test"):
        '''
        constructor for plotter
        '''
        self.rows = rows
        self.cols = cols
        self.iter = 0
        self.supress_title = supress_title
        self.is_end = False
        self.keyboard_callback = None
        self.__dyna_init()

    def set_keypress_callback(self, func):
        self.keyboard_callback = func

    def __on_key_press_callback(self, event):
        '''
            keyboard press callback function
        '''
        if self.keyboard_callback is not None:
            self.keyboard_callback(event)
        if event.key == "escape":
            self.is_end = True
            log_print("keyboard escape captured, ending...")

    def __connect(self):
        '''
            matplotlib, figure callback connection
        '''
        self.fig.canvas.mpl_connect('key_press_event',
                                    self.__on_key_press_callback)

    def __dyna_init(self):
        '''
            init the dyna plotter
        '''
        plt.ion()
        self.fig = plt.figure(self.supress_title)
        self._clear()

        # connect callback
        self.__connect()

    def _clear(self):
        self.iter = 0
        self.fig.clf()

    def add(self, image, title="hello_subplot"):
        self.iter += 1
        ax = self.fig.add_subplot(self.rows, self.cols, self.iter)
        ax.imshow(image)
        ax.title.set_text(title)

    def show(self, dt=3e-2):
        plt.pause(dt)
        self._clear()

    # def wait_for_end_key(self):


def calculate_subplot_size(num_of_images):
    row_size = 1
    col_size = None
    while col_size is not None:
        for i in range(0, max(int(row_size * 0.5), 1)):
            if row_size * (row_size + i) >= num_of_images:
                col_size = row_size + i
                break
    return row_size, col_size


if __name__ == "__main__":
    import numpy as np
    rows = 2
    cols = 2
    plotter = DynaPlotter(rows, cols)

    while plotter.is_end == False:
        img_size = 100
        image = np.random.rand(img_size, img_size)
        plotter.add(image, "the 1st image")
        plotter.add(image, "the 2nd image")
        plotter.add(image, "the 3rd image")
        plotter.show()