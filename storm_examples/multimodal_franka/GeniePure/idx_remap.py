import numpy as np
from typing import List, Optional


class JointNameMapper:
    """
    将一个来源的关节顺序（src_names）映射到统一的内部顺序（dst_names），
    支持对关节值的快速重排、缺失补全与 fallback 替代。
    """

    def __init__(self, src_names: List[str], dst_names: List[str]):
        """
        Args:
            src_names: 来源系统的关节名顺序（如 Gym、TracIK 等）
            dst_names: 内部标准顺序（如 URDF 控制器所用）
        """
        self.src_names = src_names
        self.dst_names = dst_names
        self.num_dof = len(dst_names)

        # 名称 → 索引映射（src 侧）
        self._src_name_to_idx = {name: i for i, name in enumerate(src_names)}
        # 每个 dst_name 对应的 src_q 中索引，若缺失则为 -1
        self._reorder_indices = [self._src_name_to_idx.get(name, -1) for name in dst_names]

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
        """
        print(f"{'dst_idx':>7s} | {'dst_name':<20s} | {'src_idx':>7s} | {'src_name':<20s}")
        print("-" * 60)
        for dst_idx, (dst_name, src_idx) in enumerate(zip(self.dst_names, self._reorder_indices)):
            src_name = self.src_names[src_idx] if src_idx != -1 else '---'
            src_idx_str = f"{src_idx}" if src_idx != -1 else '  -'
            print(f"{dst_idx:7d} | {dst_name:<20s} | {src_idx_str:7s} | {src_name:<20s}")


if __name__ == '__main__':
    internal_joint_names = [
        'joint_lift_body', 'joint_body_pitch',
        'joint_head_yaw', 'joint_head_pitch',
        'Joint1_l', 'Joint2_l', 'Joint3_l',
        'Joint4_l', 'Joint5_l', 'Joint6_l', 'Joint7_l'
    ]

    gym_joint_names = [
        'joint_lift_body', 'joint_body_pitch',
        'Joint1_l', 'Joint2_l', 'Joint3_l',
        'Joint4_l', 'Joint5_l', 'Joint6_l', 'Joint7_l',
        'joint_head_yaw', 'joint_head_pitch'
    ]
    q_gym = np.array([0.1 * i for i in range(len(gym_joint_names))])

    print("== Gym → Internal ==")
    gym_mapper = JointNameMapper(gym_joint_names, internal_joint_names)
    q_internal = gym_mapper.forward(q_gym)
    gym_mapper.debug_mapping()

    for i, (name, val) in enumerate(zip(internal_joint_names, q_internal)):
        print(f"{i:2d} | {name:<20s} | q_internal: {val:8.3f}")

    print("\n== IK → Internal with fallback ==")
    ik_joint_names = [
        'joint_lift_body', 'joint_body_pitch',
        'Joint1_l', 'Joint2_l', 'Joint3_l',
        'Joint4_l', 'Joint5_l', 'Joint6_l', 'Joint7_l'
    ]
    q_ik = np.array([-0.1 * i for i in range(len(ik_joint_names))])
    ik_mapper = JointNameMapper(ik_joint_names, internal_joint_names)
    q_full = ik_mapper.forward(q_ik, fallback=q_internal)

    for i, (name, val) in enumerate(zip(internal_joint_names, q_full)):
        flag = "← from IK" if name in ik_joint_names else "← fallback"
        print(f"{i:2d} | {name:<20s} | q_full: {val:8.3f} {flag}")
