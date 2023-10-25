    #     # point cloud debug
    #     # 一、 first_run
    #     voxel_collision_cost = mpc_control.controller.rollout_fn.voxel_collision_cost

    #     # voxel_collision_cost.first_run(camera_data)
    #     # 0 set world transforms:
    #     q = camera_data['robot_camera_pose'].r
    #     p = camera_data['robot_camera_pose'].p
    #     rot = quaternion_to_matrix(torch.as_tensor([q.w,q.x,q.y,q.z]).unsqueeze(0)) 

    #     robot_camera_trans = torch.tensor([p.x,p.y,p.z]).unsqueeze(0)
    #     robot_camera_rot = torch.tensor(rot)

    #     robot_table_trans = torch.tensor([0.0,-0.35,-0.24]).unsqueeze(0)
    #     robot_table_rot = torch.eye(3).unsqueeze(0)

    #     voxel_collision_cost.coll.set_world_transform(robot_table_trans, robot_table_rot,
    #                                     robot_camera_trans, robot_camera_rot)

    #     #1 voxel_collision_cost.coll.set_scene(camera_data['pc'], camera_data['pc_seg'])
    #         #1.1 self.world.update_world_pc(camera_pointcloud, scene_labels) or 
    #         # mpc_control.controller.rollout_fn.voxel_collision_cost.coll.world.update_world_pc(pointcloud, seg_labels)
    #             # 1.1.2 Fill pointcloud in tensor:
    #     pointcloud = camera_data['pc']
    #     seg_labels = camera_data['pc_seg']
    #     orig_scene_pc = torch.as_tensor(pointcloud, **tensor_args)
    #     # filter seg labels:
    #     scene_labels = torch.as_tensor(seg_labels.astype(int), **tensor_args)

    #     scene_pc_mask = torch.logical_and(scene_labels != camera_data['label_map']["robot"],
    #                                         scene_labels !=  camera_data['label_map']["ground"])

    #     vis_mask = scene_pc_mask.flatten()
    #     scene_pc = orig_scene_pc[vis_mask]

    #     scene_pc = voxel_collision_cost.coll.world.camera_transform.transform_point(scene_pc)
    # # min_bound = voxel_collision_cost.coll.world.bounds[0]
    # # max_bound = voxel_collision_cost.coll.world.bounds[1]
    # # mask_bound = torch.logical_and(torch.all(scene_pc > min_bound, dim=-1), torch.all(scene_pc < max_bound, dim=-1))

    
    # # scene_pc = scene_pc[mask_bound]
    # voxel_collision_cost.coll.world.scene_pc = scene_pc

    # # todo: try to visualize the scene pc
    
    #     #1.2 self.world.update_world_sdf(self.world.scene_pc)
    # # voxel_collision_cost.coll.world.update_world_sdf(voxel_collision_cost.coll.world.scene_pc)

    # # voxel_collision_cost.COLL_INIT = True


    # # pointcloud = pc
    # # seg_labels = camera_data['pc_seg']
    # # mpc_control.controller.rollout_fn.voxel_collision_cost.coll.world.update_world_pc(pointcloud, seg_labels)