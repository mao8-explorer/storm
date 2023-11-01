import torch
from ..cost import DistCost, ZeroCost, StopCost, PoseCostQuaternion, PoseCost_Reward, JnqSparseReward,CartSparseReward,BoundCost, PrimitiveCollisionCost,RobotSelfCollisionCost
from ..model import URDFKinematicModel
from ...util_file import join_path, get_assets_path
from ...differentiable_robot_model.coordinate_transform import matrix_to_quaternion, quaternion_to_matrix
from ...mpc.model.integration_utils import build_fd_matrix
from ...mpc.rollout.rollout_base import RolloutBase

class ArmReacherRealMultiModal(RolloutBase):
    """
    This rollout function is for reaching a cartesian pose for a robot

    Todo: 
    1. Update exp_params to be kwargs
    """

    def __init__(self, exp_params, tensor_args={'device':"cpu", 'dtype':torch.float32}, world_params=None):
        self.tensor_args = tensor_args
        self.exp_params = exp_params
        device = self.tensor_args['device']
        float_dtype = self.tensor_args['dtype']
        mppi_params = exp_params['mppi']
        model_params = exp_params['model']
        robot_params = exp_params['robot_params']
        assets_path = get_assets_path()
        #print('EE LINK',exp_params['model']['ee_link_name'])
        # initialize dynamics model:
        dynamics_horizon = mppi_params['horizon'] * model_params['dt']
        #Create the dynamical system used for rollouts
        self.dynamics_model = URDFKinematicModel(join_path(assets_path,exp_params['model']['urdf_path']),
                                                 dt=exp_params['model']['dt'],
                                                 batch_size=mppi_params['num_particles'],
                                                 horizon=dynamics_horizon,
                                                 tensor_args=self.tensor_args,
                                                 ee_link_name=exp_params['model']['ee_link_name'],
                                                 link_names=exp_params['model']['link_names'],
                                                 dt_traj_params=exp_params['model']['dt_traj_params'],
                                                 control_space=exp_params['control_space'],
                                                 vel_scale=exp_params['model']['vel_scale'])
        self.dt = self.dynamics_model.dt
        self.n_dofs = self.dynamics_model.n_dofs
        self.traj_dt = self.dynamics_model.traj_dt # trajectory dt (时变)
        self._fd_matrix_sphere = self.dynamics_model._fd_matrix_sphere # 差分矩阵 sphere_pos -> vel
    
        self.goal_state = None
        self.goal_ee_pos = None
        self.goal_ee_rot = None
        self.goal_jnq = None 
        self.curr_ee_pos = None
        self.link_pos_seq = torch.zeros((1, 1, len(self.dynamics_model.link_names), 3), **self.tensor_args)
        self.link_rot_seq = torch.zeros((1, 1, len(self.dynamics_model.link_names), 3, 3), **self.tensor_args)
        
        self.jnq_dist_cost = DistCost(**self.exp_params['cost']['joint_l2'], device=device,float_dtype=float_dtype) # Joint Space target

        self.goal_cost_reward = PoseCost_Reward(**exp_params['cost']['PoseCost_Reward'], # Cartesian space target
                                  tensor_args=self.tensor_args)
        
        self.jnq_sparse_reward = JnqSparseReward(**exp_params['cost']['Jnq_sparse_reward'], # Joint Space Reward
                                  tensor_args=self.tensor_args)

        self.primitive_collision_cost = PrimitiveCollisionCost(world_params=world_params, robot_params=robot_params, 
                                                            tensor_args=self.tensor_args, 
                                                            **self.exp_params['cost']['primitive_collision'],
                                                            traj_dt=self.traj_dt,
                                                            _fd_matrix_sphere = self._fd_matrix_sphere)

        # Safe dynamic model
        self.robot_self_collision_cost = RobotSelfCollisionCost(robot_params=robot_params, tensor_args=self.tensor_args, **self.exp_params['cost']['robot_self_collision'])
        # 处理终点震荡  zero_vel_cost + stop_cost
        self.zero_vel_cost = ZeroCost(device=device, float_dtype=float_dtype, **exp_params['cost']['zero_vel']) 
        self.stop_cost = StopCost(**exp_params['cost']['stop_cost'], 
                                  tensor_args=self.tensor_args,
                                  traj_dt=self.traj_dt)
        bounds = torch.cat([self.dynamics_model.state_lower_bounds[:self.n_dofs * 3].unsqueeze(0),self.dynamics_model.state_upper_bounds[:self.n_dofs * 3].unsqueeze(0)], dim=0).T
        self.bound_cost = BoundCost(**exp_params['cost']['state_bound'],
                                    tensor_args=self.tensor_args,
                                    bounds=bounds)


        multimodal_mppi_costs = exp_params['multimodal_cost']
        self.multiTargetCost = multimodal_mppi_costs['jnq_goal_cost']
        self.multiCollisionCost = multimodal_mppi_costs['environment_collision']
        self.multiTerminalCost = multimodal_mppi_costs['Cart_jnq_goal_reward']
        self.multiTargetCost_Cart = multimodal_mppi_costs['Cart_goal_cost']
        # 权重归1 重新分配 !
        self.jnq_dist_cost.weight = torch.tensor(1.0, device=device, dtype=float_dtype)
        self.primitive_collision_cost.weight = torch.tensor(1.0, device=device, dtype=float_dtype)
        self.jnq_sparse_reward.weight = torch.tensor(1.0, device=device, dtype=float_dtype)
        self.goal_cost_reward.pose_weight = torch.tensor(1.0, device=device, dtype=float_dtype)
        self.goal_cost_reward.reward_weight = torch.tensor(1.0, device=device, dtype=float_dtype)

        self.targetcost_greedy_w = self.multiTargetCost['greedy_weight']
        self.collision_greedy_w  = self.multiCollisionCost['greedy_weight']
        self.terminalsparse_greedy_w = self.multiTerminalCost['greedy_weight']
        self.cart_targetcost_greedy_w = self.multiTargetCost_Cart['greedy_weight']

        self.targetcost_sensi_w = self.multiTargetCost['sensi_weight']
        self.collision_sensi_w  = self.multiCollisionCost['sensi_weight']
        self.terminalsparse_sensi_w = self.multiTerminalCost['sensi_weight']
        self.cart_targetcost_sensi_w = self.multiTargetCost_Cart['sensi_weight']

        self.targetcost_judge_w = self.multiTargetCost['judge_weight']
        self.collision_judge_w  = self.multiCollisionCost['judge_weight']
        self.terminalsparse_judge_w = self.multiTerminalCost['judge_weight']
        self.cart_targetcost_judge_w = self.multiTargetCost_Cart['judge_weight']



    
    def multimodal_cost_fn(self, state_dict):
        state_batch = state_dict['state_seq']
        ee_pos_batch = state_dict['ee_pos_seq']
        link_pos_batch, link_rot_batch = state_dict['link_pos_seq'], state_dict['link_rot_seq']
        goal_ee_pos = self.goal_ee_pos



        self.bound_contraint = self.bound_cost.forward(state_batch[:,:,:self.n_dofs * 3])
        self.vel_cost = self.stop_cost.forward(state_batch[:, :, self.n_dofs:self.n_dofs * 2])
        self.robot_collision = self.robot_self_collision_cost.forward(state_batch[:,:,:self.n_dofs])
        self.environment_collision , self.judge_environment_collision= self.primitive_collision_cost.optimal_forward(link_pos_batch, link_rot_batch)

        # 为什么要存在 因为逆解不存在时，也就是全局规划无解时，可以使用该方式引导
        self.cart_goal_cost, self.cart_goal_reward = self.goal_cost_reward.forward(ee_pos_batch, goal_ee_pos)

        if self.goal_jnq is not None:
            disp_vec = state_batch[:,:,0:self.n_dofs] - self.goal_jnq[:,0:self.n_dofs]
            self.jnq_goal_cost = self.jnq_dist_cost.forward(disp_vec)
            self.jnq_goal_reward = self.jnq_sparse_reward.forward(disp_vec)
            self.zero_vel_bound = self.zero_vel_cost.forward(state_batch[:, :, self.n_dofs:self.n_dofs*2], goal_dist=disp_vec)


        # get some special indexes to visualization
        # -1 mean_action
        # 0 sensi_best_action
        # 1 greedy_best_action
        # 2 sensi_mean
        # 3 greedy_mean
        self.top_trajs = torch.cat([ee_pos_batch[-1].unsqueeze(0),ee_pos_batch[0:4]])
        self.curr_ee_pos = ee_pos_batch[-1,0,:]



    def multimodal_rollout_fn(self, start_state, act_seq):

        state_dict = self.dynamics_model.rollout_open_loop(start_state, act_seq)
        # self.multimodal_cost_fn(state_dict)
        state_batch = state_dict['state_seq']
        ee_pos_batch = state_dict['ee_pos_seq']
        link_pos_batch, link_rot_batch = state_dict['link_pos_seq'], state_dict['link_rot_seq']
        goal_ee_pos = self.goal_ee_pos

        # 基础元素计算 单兵列阵
        self.bound_contraint = self.bound_cost.forward(state_batch[:,:,:self.n_dofs * 3])
        self.vel_cost = self.stop_cost.forward(state_batch[:, :, self.n_dofs:self.n_dofs * 2])
        self.robot_collision = self.robot_self_collision_cost.forward(state_batch[:,:,:self.n_dofs])
        self.environment_collision , self.judge_environment_collision= self.primitive_collision_cost.optimal_forward(link_pos_batch, link_rot_batch)

        # 为什么要存在 因为逆解不存在时，也就是全局规划无解时，可以使用该方式引导
        self.cart_goal_cost, self.cart_goal_reward = self.goal_cost_reward.forward(ee_pos_batch, goal_ee_pos)

        if self.goal_jnq is not None:
            disp_vec = state_batch[:,:,0:self.n_dofs] - self.goal_jnq[:,0:self.n_dofs]
            self.jnq_goal_cost = self.jnq_dist_cost.forward(disp_vec)
            self.jnq_goal_reward = self.jnq_sparse_reward.forward(disp_vec)
            self.zero_vel_bound = self.zero_vel_cost.forward(state_batch[:, :, self.n_dofs:self.n_dofs*2], goal_dist=disp_vec)

        # for visualization and goal_ee_pos measure
        self.top_trajs = torch.cat([ee_pos_batch[-1].unsqueeze(0),ee_pos_batch[0:4]])
        self.curr_ee_pos = ee_pos_batch[-1,0,:]

        self.normal_cost = self.bound_contraint + self.vel_cost + self.robot_collision 

        """
        why sensi no reward cost ?
            for dynaminc avoidance near goal region
        """
        if self.goal_jnq is not None:
            greedy_cost_seq = self.jnq_goal_cost * self.targetcost_greedy_w +\
                            self.environment_collision * self.collision_greedy_w +\
                            (self.cart_goal_reward + self.jnq_goal_reward) * self.terminalsparse_greedy_w+\
                            self.cart_goal_cost * self.cart_targetcost_greedy_w +\
                            self.normal_cost   + self.zero_vel_bound 
            
            sensi_cost_seq =  self.jnq_goal_cost * self.targetcost_sensi_w +\
                            (self.cart_goal_reward + self.jnq_goal_reward) * self.terminalsparse_sensi_w+\
                            self.environment_collision * self.collision_sensi_w  +\
                            self.normal_cost
            
            judge_cost_seq = self.jnq_goal_cost * self.targetcost_judge_w+\
                             self.cart_goal_cost * self.cart_targetcost_judge_w +\
                            self.judge_environment_collision * self.collision_judge_w  +\
                            (self.cart_goal_reward + self.jnq_goal_reward) * self.terminalsparse_judge_w 
            
        else:
            greedy_cost_seq = \
                            self.environment_collision * self.collision_greedy_w +\
                            self.cart_goal_cost * self.cart_targetcost_greedy_w +\
                            self.cart_goal_reward * self.terminalsparse_greedy_w +\
                            self.normal_cost 
            
            sensi_cost_seq = \
                            self.cart_goal_cost * self.cart_targetcost_sensi_w  +\
                            self.environment_collision * self.collision_sensi_w  +\
                            self.normal_cost 
            
            judge_cost_seq = \
                            self.judge_environment_collision * self.collision_judge_w  +\
                            self.cart_goal_cost * self.cart_targetcost_judge_w +\
                            self.cart_goal_reward * self.terminalsparse_judge_w  
            
        sim_trajs = dict(
            actions=act_seq,#.clone(),
            greedy_costs=greedy_cost_seq,#clone(),
            sensi_costs=sensi_cost_seq,
            judge_costs=judge_cost_seq,
            rollout_time=0.0,
            state_seq=state_dict['ee_pos_seq']
        )
        return sim_trajs        

    def rollout_fn(self, start_state, act_seq):
        """
        Return sequence of costs and states encountered
        by simulating a batch of action sequences

        Parameters
        ----------
        action_seq: torch.Tensor [num_particles, horizon, d_act]
        """
        # rollout_start_time = time.time()
        #print("computing rollout")
        #print(act_seq)
        #print('step...')

        state_dict = self.dynamics_model.rollout_open_loop(start_state, act_seq)
        cost_seq = self.cost_fn(state_dict, act_seq)

        sim_trajs = dict(
            actions=act_seq,#.clone(),
            costs=cost_seq,#clone(),
            ee_pos_seq=state_dict['ee_pos_seq'],#.clone(),
            #link_pos_seq=link_pos_seq,
            #link_rot_seq=link_rot_seq,
            rollout_time=0.0
        )
        
        return sim_trajs

    
    def __call__(self, start_state, act_seq):
        return self.rollout_fn(start_state, act_seq)
    
    def get_ee_pose(self, current_state):
        current_state = current_state.to(**self.tensor_args)
         
        
        # ee_pos_batch, ee_rot_batch, lin_jac_batch, ang_jac_batch = self.dynamics_model.robot_model. \
        #     compute_fk_and_jacobian(current_state[:,:self.dynamics_model.n_dofs], current_state[:, self.dynamics_model.n_dofs: self.dynamics_model.n_dofs * 2], self.exp_params['model']['ee_link_name'])
        ee_pos_batch, ee_rot_batch = self.dynamics_model.robot_model.compute_fk(current_state[:,:self.dynamics_model.n_dofs], 
                                                                                             current_state[:, self.dynamics_model.n_dofs: self.dynamics_model.n_dofs * 2], 
                                                                                             self.exp_params['model']['ee_link_name'])

        ee_quat = matrix_to_quaternion(ee_rot_batch)
        state = {'ee_pos_seq':ee_pos_batch, 'ee_rot_seq':ee_rot_batch,
                #  'lin_jac_seq': lin_jac_batch, 'ang_jac_seq': ang_jac_batch,
                 'ee_quat_seq':ee_quat}
        return state
    def current_cost(self, current_state, no_coll=True):
        current_state = current_state.to(**self.tensor_args)
        
        curr_batch_size = 1
        num_traj_points = 1 #self.dynamics_model.num_traj_points
        
        # ee_pos_batch, ee_rot_batch, lin_jac_batch, ang_jac_batch = self.dynamics_model.robot_model.compute_fk_and_jacobian(current_state[:,:self.dynamics_model.n_dofs], current_state[:, self.dynamics_model.n_dofs: self.dynamics_model.n_dofs * 2], self.exp_params['model']['ee_link_name'])
        ee_pos_batch, ee_rot_batch = self.dynamics_model.robot_model.compute_fk(current_state[:,:self.dynamics_model.n_dofs], 
                                                                                             current_state[:, self.dynamics_model.n_dofs: self.dynamics_model.n_dofs * 2], 
                                                                                             self.exp_params['model']['ee_link_name'])


        link_pos_seq = self.link_pos_seq
        
        link_rot_seq = self.link_rot_seq

        # get link poses:
        for ki,k in enumerate(self.dynamics_model.link_names):
            link_pos, link_rot = self.dynamics_model.robot_model.get_link_pose(k)
            link_pos_seq[:,:,ki,:] = link_pos.view((curr_batch_size, num_traj_points,3))
            link_rot_seq[:,:,ki,:,:] = link_rot.view((curr_batch_size, num_traj_points,3,3))
            
        if(len(current_state.shape) == 2):
            current_state = current_state.unsqueeze(0)
            ee_pos_batch = ee_pos_batch.unsqueeze(0)
            ee_rot_batch = ee_rot_batch.unsqueeze(0)
            # lin_jac_batch = lin_jac_batch.unsqueeze(0)
            # ang_jac_batch = ang_jac_batch.unsqueeze(0)

        state_dict = {'ee_pos_seq':ee_pos_batch, 'ee_rot_seq':ee_rot_batch,
                    #   'lin_jac_seq': lin_jac_batch, 'ang_jac_seq': ang_jac_batch,
                      'state_seq': current_state,'link_pos_seq':link_pos_seq,
                      'link_rot_seq':link_rot_seq,
                      'prev_state_seq':current_state}
        
        cost = self.cost_fn(state_dict, None,no_coll=no_coll, horizon_cost=False, return_dist=True)

        return cost, state_dict

    def update_params(self, retract_state=None, goal_state=None, goal_ee_pos=None, goal_ee_rot=None, goal_ee_quat=None):
        """
        Update params for the cost terms and dynamics model.
        goal_state: n_dofs
        goal_ee_pos: 3
        goal_ee_rot: 3,3
        goal_ee_quat: 4

        """
        
        if(goal_ee_pos is not None):
            self.goal_ee_pos = torch.as_tensor(goal_ee_pos, **self.tensor_args).unsqueeze(0)
            self.goal_state = None
        if(goal_ee_rot is not None):
            self.goal_ee_rot = torch.as_tensor(goal_ee_rot, **self.tensor_args).unsqueeze(0)
            self.goal_ee_quat = matrix_to_quaternion(self.goal_ee_rot)
            self.goal_state = None
        if(goal_ee_quat is not None):
            self.goal_ee_quat = torch.as_tensor(goal_ee_quat, **self.tensor_args).unsqueeze(0)
            self.goal_ee_rot = quaternion_to_matrix(self.goal_ee_quat)
            self.goal_state = None
        if(goal_state is not None):
            self.goal_state = torch.as_tensor(goal_state, **self.tensor_args).unsqueeze(0)
            self.goal_ee_pos, self.goal_ee_rot = self.dynamics_model.robot_model.compute_forward_kinematics(self.goal_state[:,0:self.n_dofs], 
                                            self.goal_state[:,self.n_dofs:2*self.n_dofs], link_name=self.exp_params['model']['ee_link_name'])
            self.goal_ee_quat = matrix_to_quaternion(self.goal_ee_rot)
        
        return True
    
