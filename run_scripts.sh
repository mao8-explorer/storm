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


# 我们要做多组实验：
# 这里先做一个 浅陋的实验
python -u "/home/zm/MotionPolicyNetworks/storm_ws/storm/storm_examples/Larger_benchmark_PM_scatter_left.py"