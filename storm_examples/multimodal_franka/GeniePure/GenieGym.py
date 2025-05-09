import yaml
import numpy as np
from storm_kit.gym.core import Gym
from storm_kit.util_file import get_gym_configs_path, join_path, load_yaml, get_assets_path
from storm_kit.gym.sim_robot import RobotSim
import torch

class GenieEnv(object):
    def __init__(self, gym_instance):
        super().__init__()
        self.gym_instance = gym_instance
        self.device = torch.device('cuda', 0)
        self.tensor_args = {'device': self.device, 'dtype': torch.float32}
        self.gym = gym_instance.gym
        self.sim = gym_instance.sim
        self.env_ptr = gym_instance.env_list[0]
        self.viewer = gym_instance.viewer


    def _genie_initialize_robot_simulation(self):
        """
        contains a generic robot class
            that can load a robot asset into sim and 
            gives access to robot's state and receive command_of_policy.
        """
        # Initialize the robot simulation
        robot_yml = join_path(get_gym_configs_path(), 'genie.yml')
        with open(robot_yml) as file:
            robot_params = yaml.load(file, Loader=yaml.FullLoader)
        sim_params = robot_params['sim_params']  # get from -->'/home/zm/MotionPolicyNetworks/storm_ws/src/storm/content/configs/gym/franka.yml'
        sim_params['asset_root'] = get_assets_path()
        sim_params['collision_model']=None
        robot_pose = sim_params['robot_pose']  # robot_pose: [0, 0.0, 0, -0.707107, 0.0, 0.0, 0.707107]'
        # create robot simulation: contains a generic robot class that can load a robot asset into sim and gives access to robot's state and control.
        self.robot_sim = RobotSim(
            gym_instance=self.gym_instance.gym, 
            sim_instance=self.gym_instance.sim,
            env_instance = self.gym_instance.env_list[0],
            viewer = self.gym_instance.viewer,
            **sim_params,
            device=torch.device('cuda', 0) )
        # create gym environment: 
        self.robot_ptr = self.robot_sim.spawn_robot(self.gym_instance.env_list[0], robot_pose, coll_id=2)


class RobotSimulator(GenieEnv):
    def __init__(self, gym_instance):
        super().__init__(gym_instance = gym_instance)
        # self._environment_init()
        self._genie_initialize_robot_simulation()


    def run(self):
        while True:
            # 正常循环主体
            self.gym_instance.step()
            self.gym_instance.clear_lines()


if __name__ == '__main__':

    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    sim_params = load_yaml(join_path(get_gym_configs_path(), 'physx.yml'))
    sim_params['headless'] = False
    gym_instance = Gym(**sim_params)

    simulator = RobotSimulator(gym_instance)
    
    simulator.run()
