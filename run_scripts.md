#!/bin/bash

# 第一个命令
python -u "/home/zm/MotionPolicyNetworks/storm_ws/storm/storm_examples/benchmark_cost_scatter_left.py"

# 第二个命令
python -u "/home/zm/MotionPolicyNetworks/storm_ws/storm/storm_examples/benchmark_cost_scatter_up.py"

# python -u "/home/zm/MotionPolicyNetworks/storm_ws/storm/storm_examples/benchmark_beta_2D_left.py"
# python -u "/home/zm/MotionPolicyNetworks/storm_ws/storm/storm_examples/benchmark_beta_2D_up.py"


# 轻微扰动PM 生成的结果
python -u "/home/zm/MotionPolicyNetworks/storm_ws/storm/storm_examples/benchmark_PM_scatter_left.py"
python -u "/home/zm/MotionPolicyNetworks/storm_ws/storm/storm_examples/benchmark_PM_scatter_up.py"
# 与SM一样的采样边界 生成的结果
python -u "/home/zm/MotionPolicyNetworks/storm_ws/storm/storm_examples/Larger_benchmark_PM_scatter_left.py"
python -u "/home/zm/MotionPolicyNetworks/storm_ws/storm/storm_examples/Larger_benchmark_PM_scatter_up.py"


## P PV散点比较分析
SM 结果已经生成，较好的满足要求。
同时，可以在较大步长的情况下生成 P PV对应的实验结果
调整image_moveable_collision_cost
P 调整后 修改数据文件保存的地址！ 
# 第一个命令
1. python -u "/home/zm/MotionPolicyNetworks/storm_ws/storm/storm_examples/benchmark_cost_scatter_left.py"
# 第二个命令
2. python -u "/home/zm/MotionPolicyNetworks/storm_ws/storm/storm_examples/benchmark_cost_scatter_up.py"
PV 调整后
PV 调整后 修改数据文件保存的地址！ 
# 第一个命令
1. python -u "/home/zm/MotionPolicyNetworks/storm_ws/storm/storm_examples/benchmark_cost_scatter_left.py"
# 第二个命令
2. python -u "/home/zm/MotionPolicyNetworks/storm_ws/storm/storm_examples/benchmark_cost_scatter_up.py"


## 同一个采样边界下的数据对比（可做）
# 我们要做多组实验：
# 这里先做一个 浅陋的实验
也就是left的数据 要收集多组，一个是judge_policy_coll 采用greedy_coll的，一个是judge_policy采用正常coll的。 
我们要做的主要任务是：
PM与SM一样的采样边界 生成的结果 对比
1. judge_coll 先采用normal_coll来实现， 注意文件的保存地方
python -u "/home/zm/MotionPolicyNetworks/storm_ws/storm/storm_examples/Larger_benchmark_PM_scatter_left.py"
2. judge_coll 替换成greedy_coll 生成的数据， 更改文件的保存地方
python -u "/home/zm/MotionPolicyNetworks/storm_ws/storm/storm_examples/Larger_benchmark_PM_scatter_left.py"
改完后，恢复文件更改与greedy_coll权重更改。 跑上下实验生成新的数据。
3. python -u "/home/zm/MotionPolicyNetworks/storm_ws/storm/storm_examples/Larger_benchmark_PM_scatter_up.py"
---> 数据分析比照结果 与 SM 结果进行对比。 看看分布情况。如果比较明显，则保留，如果不明显，则放弃

## beta 测试结果
下面都默认采用：在updown * 1  / left * （greedy）*1 两个实验任务下做 看具体效果（方差比较重要）
beta 数据准备
文件名字设置好
测试的数目设置好
1. 不同topK 下的比对结果： 10， 20 ， 30， 40， 60， 90
2. 采用的judge policy权重应该偏向greedy 一些  1.0 1.2 1.5 2.0
3. beta ： 0.3，0.8 1.0 2.0 5.0 10.0 20.0 and normal-weigth

在updown * 1  / left * （greedy）*1 两个实验任务下做 看具体效果（方差比较重要）
目的是想要比较不同topK下的数据反应，但是我们不知道哪一个judge policy下效果比较明显，可能sensi 也可能 greedy，因此我们也在该同等目标下尝试多测试几组。可能偏向greedy的效果更好，碰撞次数更多。当轨迹一多，normal weight可能导致的均质化结果，时间指标更大（maybe）
同时我们也不知道是哪一个beta下可能取得较好的结果，因此我们我尝试做一些数据分析。 以求的更多维度的分析测试

加入normal_weight的测试，助力起飞瞬间
normal weight 怎么求呢？
影响的点 应该是topK以及 Judge policy的权重 与beta没有关系， 关键点在MPPI.py文件


## Franka数据加入