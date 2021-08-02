from file_util import *
from drawer_util import DynaPlotter

masked_pkl = None
raw_pkl = None


def restore_image(ypos, xpos, kernel_size=10):
    print(f"restore {xpos, ypos}")
    xpos_int = int(xpos)
    ypos_int = int(ypos)
    bias = int(kernel_size / 2)
    for i in range(-bias, bias, 1):
        for j in range(-bias, bias, 1):
            if (xpos_int + i < 0) or (xpos_int + i >= masked_pkl.shape[0]) or (
                    ypos_int + j < 0) or (ypos_int + j >= masked_pkl.shape[1]):
                continue
            masked_pkl[xpos_int + i, ypos_int + j] = raw_pkl[xpos_int + i,
                                                             ypos_int + j]


def remove_image(ypos, xpos, kernel_size=10):
    print(f"remove {xpos, ypos}")
    xpos_int = int(xpos)
    ypos_int = int(ypos)
    bias = int(kernel_size / 2)
    for i in range(-bias, bias, 1):
        for j in range(-bias, bias, 1):

            if (xpos_int + i < 0) or (xpos_int + i >= masked_pkl.shape[0]) or (
                    ypos_int + j < 0) or (ypos_int + j >= masked_pkl.shape[1]):
                continue
            # print(xpos_int + i, ypos_int + j)
            masked_pkl[xpos_int + i, ypos_int + j] = 0


def mouse_press(event):
    # pos = event['xydata']
    if event.button == 1:
        restore_image(event.xdata, event.ydata)
    elif event.button == 3:
        remove_image(event.xdata, event.ydata, 100)


save = False


def keyboard_press(event):
    global save
    if event.key == 'x':
        save = True


def handle(raw_pkl_file, masked_pkl_file, save_file):
    global masked_pkl, raw_pkl, save
    masked_pkl = load_pkl(masked_pkl_file)[0]
    raw_pkl = load_pkl(raw_pkl_file)[0]
    plot = DynaPlotter(1, 1, iterative_mode=True)
    plot.set_supresstitle(raw_pkl_file)
    plot.set_mousepress_callback(mouse_press)
    plot.set_keypress_callback(keyboard_press)
    while plot.is_end == False:

        if save == True:
            save_pkl(save_file, [masked_pkl])
            save = False

        plot.add(masked_pkl)
        try:
            plot.show()
        except:
            break


if __name__ == "__main__":
    masked_dir = "no_background_dir.log"
    raw_dir = "fixed_cutted_dir.log"
    masked_files = get_subfiles(masked_dir)
    raw_files = get_subfiles(raw_dir)

    assert len(masked_files) == len(raw_files)
    for _idx in range(0, len(masked_files)):
        masked_file = os.path.join(masked_dir, masked_files[_idx])
        raw_file = os.path.join(raw_dir, raw_files[_idx])

        handle(raw_file, masked_file, f"{_idx}.pkl")
    # raw_pkl_file = "fixed_cutted_dir.log/0-cutted-fixed.pkl"
    # masked_pkl_file = "no_background_dir.log/0-cutted-fixed-masked.pkl"
