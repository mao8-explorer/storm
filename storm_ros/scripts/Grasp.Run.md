# 运行抓取任务实现步骤
* 工控机iqr(172.16.0.1)franka控制交互
```bash
# 上位机连接NUC工控机
ssh iqr@iqr-NUC11PAHi5 
cd /home/iqr/Workspace/zeman/lowlevel_control
su
. ./devel/setup.bash
# 初始位姿定形
roslaunch franka_motion_control move_to_joint_config.launch
# 接收franka控制命令 返回franka运行状态
roslaunch franka_motion_control command_mode.launch 
```
* 抓取感知服务器sucro(172.16.0.30)抓取任务发布
```bash
#上位机连接抓取感知服务器
ssh sucro@sucro-MS-7E06 
#摄像机启动
cd /home/sucro/zeman/realsense_ros 
. ./devel/setup.bash
roslaunch realsense2_camera robot_d435i.launch 
#抓取识别程序
cd /home/sucro/zeman/graspnet
python graspnet-baseline/ReachGrasp_monitor.py 
```

* 上位机ovo(172.16.0.10)运行STORM

```bash
#kinect启动与机械臂滤除点云
roslaunch panda_moveit_config robot_pointfilter_kinect.launch  

 # end_link滑块控制接口 及 订阅工作空间目标位姿
cd ~/MotionPolicyNetworks/storm_ws/storm
/usr/bin/env python "/home/zm/MotionPolicyNetworks/
storm_ws/storm/storm_ros/src/nodes/goal_publishers/
grasp_interactive_marker_goal_publisher.py"

# STORM动态避障程序
python -u "/home/zm/MotionPolicyNetworks/storm_ws
/storm/storm_ros/scripts/ReacherMMM_Grasp_Jnc.py" 
```