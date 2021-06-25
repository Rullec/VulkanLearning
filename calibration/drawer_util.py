'''
    This script can capture lots of passive IR images, which can be used to do calibration
'''
import matplotlib.pyplot as plt
from log_util import *
import numpy as np


class DynaPlotter:
    '''
        Dynamic multiple image plotter based on matplotlib
    '''
    def __init__(self,
                 rows,
                 cols,
                 window_title="window_title",
                 iterative_mode=True):
        '''
        constructor for plotter
        '''
        self.rows = rows
        self.cols = cols
        self.iter = 0
        self.window_title = window_title
        self.is_end = False
        self.keyboard_callback = None
        self.mouse_press_callback = None
        self.iterative_mode = iterative_mode
        self.__dyna_init()

    def set_keypress_callback(self, func):
        self.keyboard_callback = func

    def set_mousepress_callback(self, func):
        self.mouse_press_callback = func

    def __on_key_press_callback(self, event):
        '''
            keyboard press callback function
        '''
        if self.keyboard_callback is not None:
            self.keyboard_callback(event)
        if event.key == "escape":
            self.is_end = True
            log_print("keyboard escape captured, ending...")

    def __on_mouse_press_callback(self, event):
        if self.mouse_press_callback is not None:
            self.mouse_press_callback(event)

    def __connect(self):
        '''
            matplotlib, figure callback connection
        '''
        self.fig.canvas.mpl_connect('key_press_event',
                                    self.__on_key_press_callback)
        self.fig.canvas.mpl_connect('button_press_event',
                                    self.__on_mouse_press_callback)

    def set_supresstitle(self, title):
        self.fig.suptitle(title)

    def __dyna_init(self):
        '''
            init the dyna plotter
        '''
        if self.iterative_mode == True:
            plt.ion()
        else:
            plt.ioff()
        self.fig = plt.figure(self.window_title)

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

    def add_histogram(self, value, title="helo_hist"):
        self.iter += 1
        ax = self.fig.add_subplot(self.rows, self.cols, self.iter)
        ax.hist(value)
        ax.title.set_text(title)

    def show(self, dt=3e-2):
        if self.iterative_mode == True:
            plt.pause(dt)
            self._clear()
        else:
            plt.show()
            self.iter = 0

    # def wait_for_end_key(self):


def calculate_subplot_size(num_of_images):
    row_size = 0
    col_size = None
    while col_size is None:
        row_size += 1
        for i in range(0, int(row_size * 0.5) + 1):
            if row_size * (row_size + i) >= num_of_images:
                col_size = row_size + i
                break

    return row_size, col_size


def cast_int32_to_uint8(image):
    min, max = np.min(image), np.max(image)
    # print(f"old min {min} max {max}")
    image = (image.astype(np.float) / (max - min) * 255).astype(np.uint8)
    # min, max = np.min(image), np.max(image)
    # print(f"new min {min} max {max}")
    return image


def resize(image, size=128):
    # height, width
    old_type = image.dtype
    height, width = image.shape
    mid = width / 2
    assert width % 2 == 0
    # to a square
    image = image[:, int(mid - height / 2):int(mid + height / 2)].astype(
        np.float32)
    # expand this square to
    from PIL import Image
    image = Image.fromarray(image)
    image = image.resize((size, size))
    image = np.array(image, dtype=old_type)
    return image


def to_gray(img):
    return (np.dot(img[..., :3], [0.299, 0.587, 0.114])).astype(np.uint8)


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