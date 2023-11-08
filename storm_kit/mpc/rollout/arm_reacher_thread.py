import torch
import time
from ..cost import DistCost, ZeroCost, StopCost, PoseCostQuaternion, PoseCost_Reward, JnqSparseReward,CartSparseReward,BoundCost, PrimitiveCollisionCost,RobotSelfCollisionCost, RobotSelfCollision_StopBound_Cost
from ..model import URDFKinematicModel
from ...util_file import join_path, get_assets_path
from ...differentiable_robot_model.coordinate_transform import matrix_to_quaternion, quaternion_to_matrix
from ...mpc.model.integration_utils import build_fd_matrix
from ...mpc.rollout.rollout_base import RolloutBase
from torch.autograd import profiler

class ArmReacherThread(RolloutBase):
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
        # bounds = torch.cat([self.dynamics_model.state_lower_bounds[:self.n_dofs * 3].unsqueeze(0),self.dynamics_model.state_upper_bounds[:self.n_dofs * 3].unsqueeze(0)], dim=0).T
        # self.selfcoll_stopbound_cost = RobotSelfCollision_StopBound_Cost(robot_params=robot_params, 
        #                                                                    tensor_args=self.tensor_args, 
        #                                                                    **self.exp_params['cost']['selfcoll_stop_bound'],
        #                                                                    bounds = bounds,
        #                                                                    traj_dt=self.traj_dt)

        self.stop_cost = StopCost(**exp_params['cost']['stop_cost'], 
                                  tensor_args=self.tensor_args,
                                  traj_dt=self.traj_dt)
        bounds = torch.cat([self.dynamics_model.state_lower_bounds[:self.n_dofs * 3].unsqueeze(0),self.dynamics_model.state_upper_bounds[:self.n_dofs * 3].unsqueeze(0)], dim=0).T
        self.bound_cost = BoundCost(**exp_params['cost']['state_bound'],
                                    tensor_args=self.tensor_args,
                                    bounds=bounds)


        # 处理终点震荡  zero_vel_cost + stop_cost
        self.zero_vel_cost = ZeroCost(device=device, float_dtype=float_dtype, **exp_params['cost']['zero_vel']) 

        self.fk_time_sum  = 0
        self.cost_time_sum = 0


    def rollout_fn(self, start_state, act_seq):

        state_dict = self.dynamics_model.rollout_open_loop(start_state, act_seq)
        state_batch = state_dict['state_seq']
        ee_pos_batch = state_dict['ee_pos_seq']
        link_pos_batch, link_rot_batch = state_dict['link_pos_seq'], state_dict['link_rot_seq']
        self.curr_ee_pos = ee_pos_batch[-1,0,:]
        goal_ee_pos = self.goal_ee_pos

        self.bound_contraint = self.bound_cost.forward(state_batch[:,:,:self.n_dofs * 3])
        self.vel_cost = self.stop_cost.forward(state_batch[:, :, self.n_dofs:self.n_dofs * 2])
        self.robot_collision = self.robot_self_collision_cost.forward(state_batch[:,:,:self.n_dofs]) 
        # self.selfcoll_stop_bound = self.selfcoll_stopbound_cost.forward(state_batch[:,:,:self.n_dofs * 3])
        self.cart_goal_cost, self.cart_sparse_reward = self.goal_cost_reward.forward(ee_pos_batch, goal_ee_pos)
        cost = self.bound_contraint + self.vel_cost + self.robot_collision + self.cart_goal_cost + self.cart_sparse_reward
        if self.exp_params['cost']['primitive_collision']['weight'] > 0:
            self.environment_collision , _ = self.primitive_collision_cost.optimal_forward(link_pos_batch, link_rot_batch)
            cost += self.environment_collision

        if self.goal_jnq is not None:
            disp_vec = state_batch[:,:,0:self.n_dofs] - self.goal_jnq[:,0:self.n_dofs]
            self.jnq_goal_cost = self.jnq_dist_cost.forward(disp_vec)
            self.jnq_goal_reward = self.jnq_sparse_reward.forward(disp_vec)
            self.zero_vel_bound = self.zero_vel_cost.forward(state_batch[:, :, self.n_dofs:self.n_dofs*2], goal_dist=disp_vec)

            cost += self.jnq_goal_cost + self.jnq_goal_reward  + self.zero_vel_bound


        #  + self.environment_collision
        sim_trajs = dict(
            actions=act_seq,
            costs=cost,
            ee_pos_seq=state_dict['ee_pos_seq'],
            rollout_time=0.0
        )

        return sim_trajs        



    def test_rollout_fn(self, start_state, act_seq):

        state_dict = self.dynamics_model.rollout_open_loop(start_state, act_seq)
        with profiler.profile(record_shapes=True, use_cuda=True) as prof:
            state_batch = state_dict['state_seq']
            ee_pos_batch = state_dict['ee_pos_seq']
            # link_pos_batch, link_rot_batch = state_dict['link_pos_seq'], state_dict['link_rot_seq']
            self.curr_ee_pos = ee_pos_batch[-1,0,:]
            goal_ee_pos = self.goal_ee_pos

            with profiler.record_function("bound_contraint"):
                self.bound_contraint = self.bound_cost.forward(state_batch[:,:,:self.n_dofs * 3])
            with profiler.record_function("vel_cost"):
                self.vel_cost = self.stop_cost.forward(state_batch[:, :, self.n_dofs:self.n_dofs * 2])
            with profiler.record_function("robot_collision"): 
                self.robot_collision = self.robot_self_collision_cost.forward(state_batch[:,:,:self.n_dofs])
            # self.environment_collision , self.judge_environment_collision= self.primitive_collision_cost.optimal_forward(link_pos_batch, link_rot_batch)
            with profiler.record_function("cart_goal_cost"): 
                self.cart_goal_cost = self.goal_cost_reward.forward(ee_pos_batch, goal_ee_pos)
            # with profiler.record_function("cart_goal_reward"): 
            #     self.cart_goal_reward = self.cart_sparse_reward.forward(ee_pos_batch,goal_ee_pos)

            cost = self.bound_contraint + self.vel_cost + self.robot_collision + self.cart_goal_cost

            if self.goal_jnq is not None:
                disp_vec = state_batch[:,:,0:self.n_dofs] - self.goal_jnq[:,0:self.n_dofs]
                with profiler.record_function("jnq_goal_cost"): 
                    self.jnq_goal_cost = self.jnq_dist_cost.forward(disp_vec)
                with profiler.record_function("jnq_goal_reward"): 
                    self.jnq_goal_reward = self.jnq_sparse_reward.forward(disp_vec)
                with profiler.record_function("zero_vel_bound"): 
                    self.zero_vel_bound = self.zero_vel_cost.forward(state_batch[:, :, self.n_dofs:self.n_dofs*2], goal_dist=disp_vec)

                cost += self.jnq_goal_cost + self.jnq_goal_reward  + self.zero_vel_bound

        print(prof.key_averages().table(sort_by="self_cpu_time_total"))

        #  + self.environment_collision
        sim_trajs = dict(
            actions=act_seq,
            costs=cost,
            ee_pos_seq=state_dict['ee_pos_seq'],
            rollout_time=0.0
        )

        return sim_trajs        

    
    def __call__(self, start_state, act_seq):
        return self.rollout_fn(start_state, act_seq)
    

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
    
