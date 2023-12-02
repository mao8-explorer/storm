# 运行抓取任务实现步骤
* 1. 工控机iqr(172.16.0.1)franka控制交互
```bash
ssh iqr@iqr-NUC11PAHi5 
cd /home/iqr/Workspace/zeman/lowlevel_control
su
. ./devel/setup.bash
roslaunch franka_motion_control move_to_joint_config.launch # 运动到初始理想位置
roslaunch franka_motion_control command_mode.launch # 接收franka控制命令 返回franka运行状态

```
* 2. 服务器sucro(172.16.0.30)抓取任务发布
```bash
ssh sucro@sucro-MS-7E06
cd /home/sucro/zeman/realsense_ros
. ./devel/setup.bash
roslaunch realsense2_camera robot_d435i.launch #摄像机启动

cd /home/sucro/zeman/graspnet
python graspnet-baseline/ReachGrasp_monitor.py #抓取识别程序
```
* 3. 本机ovo(172.16.0.10)运行STORM

```bash
roslaunch panda_moveit_config robot_pointfilter_kinect.launch  #kinect启动与机械臂滤除点云

cd ~/MotionPolicyNetworks/storm_ws/storm
/usr/bin/env python "/home/zm/MotionPolicyNetworks/storm_ws/storm/storm_ros/src/nodes/goal_publishers/grasp_interactive_marker_goal_publisher.py" # end_link滑块控制
python -u "/home/zm/MotionPolicyNetworks/storm_ws/storm/storm_ros/scripts/ReacherMMM_Grasp_Jnc.py" # STORM

```