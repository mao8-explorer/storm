import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 定义10组2D数据组成的轨迹
data = np.random.randn(10, 30, 2)*350

# 加载image图像
img = plt.imread('/home/zm/MotionPolicyNetworks/storm_ws/storm/content/assets/collision_maps/collision_map_cem.png')

# 创建绘图窗口和轨迹图层
fig = plt.figure()
ax = plt.subplot(111)
line, = ax.plot([], [], linewidth=2)

# 定义动画函数
def animate(i):
    # 获取当前帧的轨迹数据
    x = data[i, :, 0]
    y = data[i, :, 1]
    # 绘制轨迹
    line.set_data(x, y)
    # 返回轨迹图层
    return line,

# 定义初始化函数
def init():
    # 绘制背景图像
    ax.imshow(img)
    # 返回轨迹图层
    return line,

# 创建动画对象
ani = animation.FuncAnimation(fig, animate, frames=len(data), init_func=init, blit=True)

# 显示动画
plt.show()
