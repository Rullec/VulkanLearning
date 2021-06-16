from diff_depth_image import load_capture_depth_image
import video_manager
import matplotlib.pyplot as plt
import os
plt.ion()
fig1 = plt.figure('frame')

output_dir = "captured_depth_images/"
cam = video_manager.video_manager()
import shutil
if os.path.exists(output_dir) == True:
    shutil.rmtree(output_dir)
os.makedirs(output_dir)


iters = 1
while True:
    # clear but do not close the figure
    fig1.clf()
    ax1 = fig1.add_subplot(1, 1, 1)
    captured_img = load_capture_depth_image(cam)
    import pickle
    path = os.path.join(output_dir, f"{iters}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(captured_img, f)
        print(f"[debug] save to {path}")
    ax1.imshow(captured_img)
    ax1.title.set_text("captured depth image")
    plt.pause(3e-2)
    iters +=1 