import matplotlib.pyplot as plt
import torch
import numpy as np

plt.figure()
ax = plt.subplot(1,1,1)

t = torch.arange(18).view(3,-1)
t[1,:]+=2
ax.imshow(t.cpu())
grad_y,grad_x = torch.gradient(t)

y_range, x_range = t.shape # 关键
x_step = 1
y_step = 1
x = np.arange(0, x_range,x_step)
y = np.arange(0, y_range,y_step)
X, Y = np.meshgrid(x, y)

ax.quiver(X, Y,
               grad_x[::x_step, ::y_step].cpu().numpy(),
               grad_y[::x_step, ::y_step].cpu().numpy(),
               cmap=plt.cm.jet)  # quiver自己的问题 并不会随着ax.invert_yaxis()改变