import matplotlib.pyplot as plt
from pytransform3d.urdf import UrdfTransformManager
from mpl_toolkits.mplot3d import Axes3D


# 创建变换管理器的实例
tm = UrdfTransformManager()

# 从文件中加载URDF模型
with open("franka_panda_no_gripper.urdf", 'r') as urdf_file:
    urdf_string = urdf_file.read()
tm.load_urdf(urdf_string)

joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
for i,j in enumerate(joint_names):
    tm.set_joint(j, 0.5)

ax = tm.plot_frames_in(
    "base_link", s=0.1,
    show_name=False)

tm.plot_visuals("base_link", ax=ax, convex_hull_of_mesh=True)

# tm.plot_visuals("base", ax=ax, convex_hull_of_mesh=True)
plt.show()
