from typing import List, Optional
import numpy as np

class JointNameMapper:
    """
    将一个来源的关节顺序（src_names）映射到统一的内部顺序（dst_names），
    支持对关节值的快速重排、缺失补全与 fallback 替代。
    
    主要功能:
    1. 在不同模块的关节顺序之间建立映射关系
    2. 支持关节值数组的重排序
    3. 处理缺失关节的情况，提供默认值或 fallback 机制
    4. 提供调试信息打印功能
    
    使用示例:
    ```python
    # 创建映射器实例
    mapper = JointNameMapper(
        src_names=['joint1', 'joint2'],  # 源关节顺序
        dst_names=['joint2', 'joint1', 'joint3']  # 目标关节顺序
    )
    
    # 重排关节值数组
    src_q = np.array([1.0, 2.0])
    dst_q = mapper.forward(src_q, default_value=0.0)
    # 结果: array([2.0, 1.0, 0.0])
    
    # 打印映射关系
    mapper.debug_mapping()
    ```
    """

    def __init__(self, src_names: List[str], dst_names: List[str]):
        """
        初始化关节名称映射器。
        
        Args:
            src_names: 来源系统的关节名顺序（如 Gym、TracIK 等）
            dst_names: 内部标准顺序（如 URDF 控制器所用）
            
        Note:
            - src_names 和 dst_names 可以包含不同数量的关节
            - 不是所有关节都需要有对应关系
        """
        self.src_names = src_names
        self.dst_names = dst_names
        self.num_dof = len(dst_names)

        # 构建名称到索引的映射
        self._src_name_to_idx = {name: i for i, name in enumerate(src_names)}
        self._dst_name_to_idx = {name: i for i, name in enumerate(dst_names)}
        
        # 构建重排序索引
        self._reorder_indices = [
            self._src_name_to_idx.get(name, -1) for name in dst_names
        ]
        self._src_to_dst_index = [
            self._dst_name_to_idx.get(name, -1) for name in src_names
        ]

    def forward(
        self,
        src_q: np.ndarray,
        fallback: Optional[np.ndarray] = None,
        default_value: float = 0.0
    ) -> np.ndarray:
        """
        将 src_q 重排为 dst_names 顺序，缺失值自动填补。
        
        Args:
            src_q: 原始关节值数组，对应 src_names 顺序
            fallback: 缺失关节名时使用的默认值数组（如已有状态）
            default_value: 若未提供 fallback，则使用该默认值补全
            
        Returns:
            np.ndarray: 重排并补全后的关节数组，对应 dst_names 顺序
            
        Example:
            ```python
            # 假设有以下映射关系:
            # src_names = ['j1', 'j2']
            # dst_names = ['j2', 'j1', 'j3']
            
            src_q = np.array([0.1, 0.2])
            
            # 使用默认值
            dst_q = mapper.forward(src_q, default_value=0.0)
            # 结果: array([0.2, 0.1, 0.0])
            
            # 使用 fallback
            fallback = np.array([1.0, 1.0, 1.0])
            dst_q = mapper.forward(src_q, fallback=fallback)
            # 结果: array([0.2, 0.1, 1.0])
            ```
        """
        if fallback is not None:
            dst_q = fallback.copy()
        else:
            dst_q = np.full(self.num_dof, default_value, dtype=src_q.dtype)

        for dst_idx, src_idx in enumerate(self._reorder_indices):
            if src_idx != -1:
                dst_q[dst_idx] = src_q[src_idx]
        return dst_q

    def debug_mapping(self):
        """
        打印 src_names → dst_names 的映射关系，用于调试。
        
        输出格式示例:
        ```
        dst_idx | dst_name            | src_idx | src_name           
        ------------------------------------------------------------
           0    | joint2              |    1    | joint2            
           1    | joint1              |    0    | joint1            
           2    | joint3              |    -    | ---               
        ```
        """
        print(f"{'dst_idx':>7s} | {'dst_name':<20s} | {'src_idx':>7s} | {'src_name':<20s}")
        print("-" * 60)
        
        for dst_idx, (dst_name, src_idx) in enumerate(zip(self.dst_names, self._reorder_indices)):
            src_name = self.src_names[src_idx] if src_idx != -1 else '---'
            src_idx_str = f"{src_idx}" if src_idx != -1 else '  -'
            print(f"{dst_idx:7d} | {dst_name:<20s} | {src_idx_str:7s} | {src_name:<20s}")