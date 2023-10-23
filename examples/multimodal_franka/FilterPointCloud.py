import numpy as np
import trimesh.transformations as tra
import torch

np.set_printoptions(suppress=True)


class FilterPointCloud(object):
    def __init__(
        self,
        camera_pose=None,
        device=0,
    ):
        self.device = device
        self.cam_type = 0

        self.label_map = {
            "background": 0,
            "env": 1,
            "robot": 2,
            "target": 0,
            "tabel":20
        }

        self.camera_pose = tra.euler_matrix(np.pi / 2, 0, 0) @ camera_pose

    def _update_state(self, obs):
        """
        syncs up the scene with the latest observation.
        Args:
        obs: np.array, pointcloud of the scene
        state: dict, this is the gym_state_dict coming from scene managers.
          contains info about robot and object."""

        orig_scene_pc = obs["pc"][self.cam_type]
        scene_labels = obs["pc_label"][self.cam_type]

        # Remove robot points plus excluded
        scene_pc_mask = np.logical_and(
            scene_labels != self.label_map["robot"],
            scene_labels != self.label_map["target"],
        )
        # self.scene_pc_mask = scene_labels ==  label_map['env']
        # Transform into robot frame (z up)
        self.scene_pc = tra.transform_points(orig_scene_pc, self.camera_pose)[scene_pc_mask]
        self.cur_scene_pc = torch.from_numpy(self.scene_pc).float().to(self.device)

        # scene_pc = tra.transform_points(orig_scene_pc, self.camera_pose)
        # new_pc = torch.from_numpy(scene_pc).float().to(self.device)
        # vis_mask = torch.from_numpy(scene_pc_mask).to(self.device)   
        # self.cur_scene_pc = new_pc[vis_mask]


