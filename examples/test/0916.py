import matplotlib.pyplot as plt
import time
import numpy as np

import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg图形后端

pause = False

def key_call_back(event):
    global pause
    pause = not pause
    print(event)

fig = plt.figure()
ax = plt.subplot(1,1,1)

fig.canvas.mpl_connect('key_press_event', key_call_back)


a = np.ones([50,50])

while (True):
    ax.cla()
    plt.imshow(a)
    plt.pause(1)

    if pause:
        time.sleep(1)
        print(pause)
        continue

    
    
    print(pause)


