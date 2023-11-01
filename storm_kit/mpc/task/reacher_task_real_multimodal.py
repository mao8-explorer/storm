
import torch
from ...mpc.rollout import ArmReacherRealMultiModal
from .arm_task import ArmTask


class ReacherTaskRealMultiModal(ArmTask):
    """
    .. inheritance-diagram:: ReacherTask
       :parts: 1

    """
    def __init__(self, task_file='ur10.yml', world_file='collision_env.yml', tensor_args={'device':"cpu", 'dtype':torch.float32}):
        
        super().__init__(task_file=task_file,
                         world_file=world_file, tensor_args=tensor_args)

    def get_rollout_fn(self, **kwargs):
        # rollout_fn = ArmReacher(**kwargs)
        rollout_fn = ArmReacherRealMultiModal(**kwargs)
        return rollout_fn

