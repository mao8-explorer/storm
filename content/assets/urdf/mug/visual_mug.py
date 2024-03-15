import matplotlib.pyplot as plt
from pytransform3d.urdf import UrdfTransformManager
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from pytransform3d.transformations import transform_from_pq


# 给定的四元数和位置
q_mug_w = np.array([0.33402004837989807, 0.27801668643951416, 0.6680400967597961, -0.604036271572113])
p_mug_w = np.array([0.4000000059604645, 0.6000000238418579, 0])
pq = np.hstack((p_mug_w, q_mug_w))  # Position and unit quaternion for no rotation
T_mug_w = transform_from_pq(pq)
# 创建从世界坐标系到坐标系c的旋转矩阵
R_w_R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
# 创建从世界坐标系到robot的变换矩阵
T_w_R = np.eye(4)
T_w_R[:3, :3] = R_w_R
T_mug_R = np.dot(T_w_R, T_mug_w)


# 创建变换管理器的实例
tm = UrdfTransformManager()

# 从文件中加载 URDF 模型
with open("movable_collision_test.urdf", 'r') as urdf_file:
    urdf_string = urdf_file.read()
tm.load_urdf(urdf_string, mesh_path='./')

# 创建一个 figure 对象
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# 设置绘图范围
ax.set_xlim3d(-0.5, 0.7)
ax.set_ylim3d(-0.6, 0.6)
ax.set_zlim3d(0, 1)

# 添加 base 框架的坐标轴
ax = tm.plot_frames_in("base", ax=ax, s=0.3, show_name=False) # s控制坐标系大小

# 设置多个 mug 的位置和透明度
positions = [
    np.array([0.400, -0.30, 0.6]),
    np.array([0.400, 0.0, 0.6]),
    np.array([0.400, 0.3, 0.6])
]
alphas = [0.2, 0.6, 0.2]  # 设置透明度

for position, alpha in zip(positions, alphas):
    T_mug_R[:3, -1] = position
    tm.add_transform("mug", "base", T_mug_R)
    tm.plot_visuals("base", ax=ax, convex_hull_of_mesh=False, alpha=alpha)

# 显示图形
plt.show()
