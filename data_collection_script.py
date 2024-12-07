import os
import numpy as np
import time
from pynput import mouse
from myUtils.RS import RealSenseCamera

real_sense_camera = RealSenseCamera(visualization=True)
real_sense_camera.start()

objects = ['chair', 'wine_glass_top', 'wine_glass_bottom', 'cup', 'bottle', 'bowl', 'remote','bagpack']
i = 7

def on_click(x, y, button, pressed):
    if pressed:
        filename = f"scene1_{objects[i]}_{time.time()}"
        real_sense_camera.save_current_rgbd_frame(os.path.join("data", filename))
    else:
        print("Mouse Released at ({0}, {1}) with {2}".format(x, y, button))

def main():
    try:
        while True:
            with mouse.Listener(on_click=on_click) as listener:
                listener.join()
    except KeyboardInterrupt:
        print("Keyboard Interrupt Detected. Stopping the RealSense Camera.")
        real_sense_camera.stop()

if __name__ == "__main__":
    main()