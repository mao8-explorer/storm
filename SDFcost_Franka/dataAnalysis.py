import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pytransform3d.urdf import UrdfTransformManager
from pytransform3d.transformations import transform_from_pq


with open('FrankaPM20.150.LEFT0.40.pkl', 'rb') as f:
    traj_log = pickle.load(f)

plt.figure()
position = np.matrix(traj_log['position'])
vel = np.matrix(traj_log['velocity'])
acc = np.matrix(traj_log['acc'])
des = np.matrix(traj_log['des'])
weights = np.matrix(traj_log['weights'])
thresh_indexs = np.matrix(traj_log['thresh_index'])  # shape is i * M

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(8, 8))
window_size = 4  # Define the window size


joint_1 = 0
joint_2 = 2
def update(frame):
    axs[0].cla()
    axs[1].cla()
    axs[2].cla()
    # axs[3].cla()

    # axs[0].set_title('Position')
    # axs[1].set_title('Velocity')
    # axs[2].set_title('Weights')

    if frame < window_size:  # Make sure we have enough data points for a sliding window
        return
    
    thresh_index = thresh_indexs[0, frame]
    pre_index = thresh_indexs[0, frame - window_size]  # Use current frame minus window size for the sliding window
    

    # Plot data on each subplot
    axs[0].plot(position[pre_index:thresh_index, joint_1], 'r', label='joint1')
    axs[0].plot(position[pre_index:thresh_index, joint_2], 'g', label='joint3')
    axs[0].plot(des[pre_index:thresh_index, joint_1], 'r-.', label='joint1_des')
    axs[0].plot(des[pre_index:thresh_index, joint_2], 'g-.', label='joint3_des')

    axs[1].plot(vel[pre_index:thresh_index, joint_1], 'r', label='joint1')
    axs[1].plot(vel[pre_index:thresh_index, joint_2], 'g', label='joint3')

    # axs[2].plot(weights[pre_index:thresh_index, 0], 'r', label='greedy')
    axs[2].plot(weights[pre_index:thresh_index, 1], 'g', label='sensi')

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    # axs[3].legend()

    fig.suptitle(f'Frame: {frame}', fontsize=16)  # Add frame title
    plt.tight_layout()

# Create the animation
ani = FuncAnimation(fig, update, frames=range(window_size, thresh_indexs.shape[1]), interval=1000)
plt.show()



