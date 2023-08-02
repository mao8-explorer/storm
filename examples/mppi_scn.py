
import numpy as np

import torch
import trimesh.transformations as tra

from scenecollisionnet.policy.collision_checker  import (
    FCLMultiSceneCollisionChecker,
    NNSceneCollisionChecker,
)
from scenecollisionnet.policy.robot import Robot


import time

np.set_printoptions(suppress=True)


class MPPIPolicy:
    def __init__(
        self,
        scene_coll_nn="/home/zm/MotionPolicyNetworks/SceneCollisionNet/weights/scene_coll_nn",
        # scene_coll_nn=None,
        cam_type="ws",
        device=0,
    ):

        self.device = device
        if cam_type not in ["ws", "hand"]:
            raise ValueError("Invalid cam_type (ws or hand)")
        self.cam_type = 0 if cam_type == "ws" else 1

        self.robot = Robot(
            "/home/zm/MotionPolicyNetworks/SceneCollisionNet/data/panda/franka_panda_no_gripper.urdf",
            "ee_link",
            device=torch.device("cuda:{:d}".format(self.device)),
        )

        self.scene_collision_checker = (
            NNSceneCollisionChecker(
                scene_coll_nn,
                self.robot,
                device=torch.device(f"cuda:{self.device}"),
                use_knn=False,
            )
            if scene_coll_nn is not None
            else FCLMultiSceneCollisionChecker(self.robot, use_scene_pc=True)
        )


    def _update_state(self, obs):
        """
        syncs up the scene with the latest observation.
        Args:
        obs: np.array, pointcloud of the scene
        state: dict, this is the gym_state_dict coming from scene managers.
          contains info about robot and object."""
        # self.robot_q = obs["robot_q"].astype(np.float64).copy()
        # label_map = {
        #     "table": TABLE_LABEL,
        #     "target": 1,
        #     "objs": 2,
        #     "robot": ROBOT_LABEL,
        # }
        label_map = {
            "background": 0,
            "env": 1,
            "robot": 2,
            "target": 0,
            "tabel":20
        }
        rtm = np.eye(4)
        # rtm[:3, 3] = (0, 0, 0.0)
 
        in_obs = {
            "pc": obs["pc"][self.cam_type],
            "pc_label": obs["pc_label"][self.cam_type],
            "label_map": label_map,
            "camera_pose": tra.euler_matrix(np.pi / 2, 0, 0)
            @ obs["camera_pose"][self.cam_type],
            "robot_to_model": rtm,
            "model_to_robot": np.linalg.inv(rtm),
        }

        self.scene_collision_checker.set_scene(in_obs)



    # Main policy call, returns a rollout based on the current observation
    def set_scene(self, obs):

        self._update_state(obs) # syncs up the scene with the latest observation.

    # Check collisions between the robot and optionally the object in hand with the scene
    # for a batch of rollouts
    def _check_collisions(self, rollouts):

        threshold=0.45

        last_time = time.time_ns()

        colls_by_link, colls_value = self.scene_collision_checker(rollouts, threshold=0.45,by_link=True)

        check_time = (time.time_ns() - last_time)/1e6
        # res = colls_value > threshold 


        colls_all = colls_by_link.any(dim=0)



        print(
            rollouts.reshape(-1),       
            colls_value.reshape(-1).cpu().numpy(),

            colls_by_link.reshape(-1).cpu().numpy(),        
            colls_all.cpu().numpy(),
            check_time
            )

        return colls_by_link,colls_all, colls_value
    

    def _fcl_check_collisions(self,rollouts):


        colls_by_link = self.scene_collision_checker(rollouts, threshold=0.45,by_link=True)

        print(colls_by_link.squeeze().reshape(-1).cpu().numpy())

    
    
    


