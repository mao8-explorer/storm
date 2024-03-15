import pickle
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
with open('visual_traj.pkl', 'rb') as f:
    traj_log = pickle.load(f)

position = np.matrix(traj_log['position'])
vel = np.matrix(traj_log['velocity'])
acc = np.matrix(traj_log['acc'])
des = np.matrix(traj_log['des'])
weights = np.matrix(traj_log['weights'])
thresh_indexs = np.matrix(traj_log['thresh_index'])  # shape is i * M

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(8, 8))




axs[0].set_title('Position')
axs[1].set_title('Velocity')
# axs[2].set_title('Acceleration')
axs[2].set_title('Weights')

window_size = 4  # Define the window size
frame = 5
thresh_index = thresh_indexs[0, frame]
pre_index = thresh_indexs[0, frame - window_size]  # Use current frame minus window size for the sliding window

# Plot data on each subplot
axs[0].plot(position[pre_index:thresh_index, 0], 'r', label='x')
axs[0].plot(position[pre_index:thresh_index, 1], 'g', label='y')
axs[0].plot(des[pre_index:thresh_index, 0], 'r-.', label='x_des')
axs[0].plot(des[pre_index:thresh_index, 1], 'g-.', label='y_des')

axs[1].plot(vel[pre_index:thresh_index, 0], 'r', label='x')
axs[1].plot(vel[pre_index:thresh_index, 1], 'g', label='y')

# axs[2].plot(acc[pre_index:thresh_index, 0], 'r', label='acc')
# axs[2].plot(acc[pre_index:thresh_index, 1], 'g', label='acc')

axs[2].plot(weights[pre_index:thresh_index, 0], 'r', label='greedy')
axs[2].plot(weights[pre_index:thresh_index, 1], 'g', label='sensi')

axs[0].legend()
axs[1].legend()
axs[2].legend()
# axs[3].legend()
plt.tight_layout()

plt.show()


