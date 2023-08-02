
```python
# 先改成control_loop() 配置当前机械臂关节状态
python "/home/zm/MotionPolicyNetworks/storm_ws/src/storm/storm_ros/src/nodes/test_reaher.py" 

# 然后 market可以发布目标点位指令
python "/home/zm/MotionPolicyNetworks/storm_ws/src/storm/storm_ros/src/nodes/test_interactive_marker_goal_publisher.py"

#然后改成control_loop_v2() 直接运行
python "/home/zm/MotionPolicyNetworks/storm_ws/src/storm/storm_ros/src/nodes/test_reaher.py" 

```
启动 仿真环境

```c++
roslaunch panda_moveit_config demo_sim.launch 
rviz
```