import copy
import numpy as np
import os


from isaacgym import gymutil, gymtorch, gymapi
import torch

from isaacgym.torch_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask


class FrankaReacher(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

    
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        num_obs = 14
        num_acts = 7

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numActions"] = num_acts


        super().__init__(
            config=self.cfg, 
            rl_device=rl_device, 
            sim_device=sim_device, 
            graphics_device_id=graphics_device_id, 
            headless=headless, 
            virtual_screen_capture=virtual_screen_capture, 
            force_render=force_render)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.franka_default_dof_pos = to_torch([1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035], device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.franka_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_franka_dofs]
        self.franka_dof_pos = self.franka_dof_state[..., 0]
        self.franka_dof_vel = self.franka_dof_state[..., 1]

        #TODO: Figure out if 13 is right
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        # if self.num_props > 0:
            # self.prop_states = self.root_state_tensor[:, 2:]

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.franka_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * (2), dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.reset_idx(torch.arange(self.num_envs, device=self.device))


    def create_sim(self):
        # implement sim set up and environment creation here
        #    - set up-axis
        #    - call super().create_sim with device args (see docstring)
        #    - create ground plane
        #    - set up environments
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../content/assets")
        franka_asset_file = "urdf/franka_description/meshes/robots/franka_panda_no_gripper.urdf"
        obj_asset_file = "urdf/mug/mug.urdf"
        target_asset_file = "urdf/mug/mug.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)
            obj_asset_file = self.cfg["env"]["asset"].get("assetFileNameObject", obj_asset_file)
            target_asset_file = self.cfg["env"]["asset"].get("assetFileNameTarget", target_asset_file)
            
        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        #load ee mug asset
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        # asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        # asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.001
        object_asset = self.gym.load_asset(self.sim, asset_root, obj_asset_file, asset_options)

        #load target mug asset
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        # asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        # asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.001
        target_asset = self.gym.load_asset(self.sim, asset_root, target_asset_file, asset_options)


        franka_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)
        self.num_obj_bodies = self.gym.get_asset_rigid_body_count(object_asset)
        self.num_obj_dofs = self.gym.get_asset_rigid_body_count(object_asset)
        self.num_target_bodies = self.gym.get_asset_rigid_body_count(target_asset)
        self.num_target_dofs = self.gym.get_asset_rigid_body_count(target_asset)
        # self.num_cabinet_bodies = self.gym.get_asset_rigid_body_count(cabinet_asset)
        # self.num_cabinet_dofs = self.gym.get_asset_dof_count(cabinet_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)
        # print("num franka bodies: ", self.num_franka_bodies)
        # print("num franka dofs: ", self.num_franka_dofs)
 
        # print("num cabinet bodies: ", self.num_cabinet_bodies)
        # print("num cabinet dofs: ", self.num_cabinet_dofs)
        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
            else:
                franka_dof_props['stiffness'][i] = 7000.0
                franka_dof_props['damping'][i] = 50.0

            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)

        # create prop assets
        # box_opts = gymapi.AssetOptions()
        # box_opts.density = 400
        # prop_asset = self.gym.create_box(self.sim, self.prop_width, self.prop_height, self.prop_width, box_opts)

        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        target_start_pose = gymapi.Transform()
        target_start_pose.p = gymapi.Vec3(0.5, 0.0, 0.5)
        target_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)


        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        num_obj_shapes = self.gym.get_asset_rigid_shape_count(object_asset)
        num_target_shapes = self.gym.get_asset_rigid_shape_count(target_asset)
        # num_cabinet_bodies = self.gym.get_asset_rigid_body_count(cabinet_asset)
        # num_cabinet_shapes = self.gym.get_asset_rigid_shape_count(cabinet_asset)
        # num_prop_bodies = self.gym.get_asset_rigid_body_count(prop_asset)
        # num_prop_shapes = self.gym.get_asset_rigid_shape_count(prop_asset)
        max_agg_bodies = num_franka_bodies + self.num_obj_bodies + self.num_target_bodies #+ self.num_props * num_prop_bodies
        max_agg_shapes = num_franka_shapes + num_obj_shapes + num_target_shapes #+ self.num_props * num_prop_shapes

        self.frankas = []
        self.objects = []
        self.targets = []
        # self.cabinets = []
        self.default_prop_states = []
        self.prop_start = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            # ee_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "ee_link")
            # ee_pose = self.gym.get_rigid_transform(env_ptr, ee_handle)
            obj_actor = self.gym.create_actor(env_ptr, object_asset, target_start_pose, "object", i, 1, 0)
            target_actor = self.gym.create_actor(env_ptr, target_asset, target_start_pose, "target", i, 1, 0)
 

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)            

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
    
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)
            self.objects.append(obj_actor)
            self.targets.append(target_actor)


        # self.ee_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_link7")
        # ee_pose = self.gym.get_rigid_transform(env_ptr, self.ee_handle)
        # print(ee_pose.p, ee_pose.r)
        self.init_data()
    
    def init_data(self):
        pass
        # for env_ptr, franka_ptr, obj_ptr in zip(self.envs, self.frankas, self.objects):
        #     ee_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_ptr, "ee_link")
        #     ee_pose = self.gym.get_rigid_transform(env_ptr, ee_handle)
        #     obj_body_ptr = self.gym.get_actor_rigid_body_handle(env_ptr, obj_ptr, 0)
        #     self.gym.set_rigid_transform(env_ptr, obj_body_ptr, copy.deepcopy(ee_pose))




    def compute_reward(self):
        pass

    def compute_observations(self):
        pass


    def pre_physics_step(self, actions):
        # implement pre-physics simulation code here
        #    - e.g. apply actions
        self.actions = actions.clone().to(self.device)
        targets = self.franka_dof_targets[:, :self.num_franka_dofs] + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.franka_dof_targets[:, :self.num_franka_dofs] = tensor_clamp(
            targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        env_ids_int32 = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.franka_dof_targets))

    def post_physics_step(self):
        # implement post-physics simulation code here
        #    - e.g. compute reward, compute observations
        pass
