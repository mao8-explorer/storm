import numpy as np
np.int = int
np.float = float

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import yaml
import torch
import copy

from urdfpy import URDF
from storm_kit.util_file import (
    join_path, get_assets_path, get_mpc_configs_path,
    get_gym_configs_path, load_yaml
)
from storm_kit.mpc.task.reacher_task import ReacherTask



# === 配置文件路径 ===
ROBOT_FILE = 'genie.yml'
TASK_FILE = 'genie_reacher.yml'
WORLD_FILE = 'collision_primitives_3d.yml'


# === Tensor 类型设定 ===
tensor_args = {'device': torch.device('cpu'), 'dtype': torch.float32}

# === 加载 YAML 配置 ===
robot_params = load_yaml(join_path(get_gym_configs_path(), ROBOT_FILE))
world_params = load_yaml(join_path(get_gym_configs_path(), WORLD_FILE))
task_params = load_yaml(join_path(get_mpc_configs_path(), TASK_FILE))

# === 加载模型并渲染 ===
urdf_path = join_path(get_assets_path(), task_params['model']['urdf_path'])
mesh_dir = join_path(get_assets_path(), 'urdf/genie_description/')

# === 初始化 MPC 控制器 ===
mpc = ReacherTask(TASK_FILE, WORLD_FILE, tensor_args)

# === 设置目标状态与当前状态 ===
goal_pos = np.array([0.1, 0.1, 0.1])
goal_quat = np.array([1.0, 0.0, 0.0, 0.0])
mpc.update_params(goal_ee_pos=goal_pos, goal_ee_quat=goal_quat)



# === 渲染函数 ===
def get_robot_meshes(urdf_path, mesh_dir, q):
    robot = URDF.load(urdf_path)
    joint_names = [j.name for j in robot.actuated_joints]
    joint_dict = dict(zip(joint_names, q))
    meshes = []
    fk = robot.link_fk(cfg=joint_dict)
    for link in robot.links:
        for visual in link.visuals:
            if visual.geometry.mesh is None:
                continue
            mesh_path = join_path(mesh_dir, visual.geometry.mesh.filename)
            try:
                m = o3d.io.read_triangle_mesh(mesh_path)
                m.compute_vertex_normals()
                m.transform(fk[link])
                m.paint_uniform_color([0.6, 0.6, 0.6])
                meshes.append(m)
            except:
                pass
    return meshes


def get_collision_spheres(q):
    state = {
        'position': q,
        'velocity': np.zeros(7),
        'acceleration': np.zeros(7)
    }
    state_arr = np.hstack((state['position'], state['velocity'], state['acceleration']))
    state_tensor = torch.as_tensor(state_arr, **tensor_args).unsqueeze(0)
    mpc.controller.rollout_fn.current_cost(state_tensor)

    pos_seq = copy.deepcopy(mpc.controller.rollout_fn.link_pos_seq)
    rot_seq = copy.deepcopy(mpc.controller.rollout_fn.link_rot_seq)

    distcheck_spheres = mpc.controller.rollout_fn.robot_self_collision_cost.distance
    dist_spheres = distcheck_spheres(pos_seq, rot_seq)[0].cpu().numpy() # shape : (1,)


    distcheck_nn = mpc.controller.rollout_fn.robot_self_collision_cost.coll.check_self_collisions_nn
    dist_nn = distcheck_nn(torch.tensor(q, device='cpu', dtype=torch.float32)).cpu().numpy()

    error = abs(dist_spheres.item() - dist_nn.item())
    print(f"Ground truth: {dist_spheres.item():.4f}, Prediction: {dist_nn.item():.4f}, Absolute error: {error:.4f}")

    b, h, n = pos_seq.shape[:3]
    pos = pos_seq.view(b * h, n, 3)
    rot = rot_seq.view(b * h, n, 3, 3)

    mpc.controller.rollout_fn.robot_self_collision_cost.coll.update_batch_robot_collision_objs(pos, rot)
    spheres = [s.numpy() for s in mpc.controller.rollout_fn.robot_self_collision_cost.coll.w_batch_link_spheres]

    meshes = []
    for sphere in spheres:
        for x, y, z, r in sphere[0]:
            m = o3d.geometry.TriangleMesh.create_sphere(radius=r)
            m.translate([x, y, z])
            m.paint_uniform_color([1, 0, 0])
            m.compute_vertex_normals()
            meshes.append(m)
    return meshes, dist_spheres, dist_nn



# === Open3D GUI 应用 ===
class RobotGUI:
    def __init__(self, q_init):
        self.q_init = q_init.copy()
        self.q = q_init.copy()
        self.window = gui.Application.instance.create_window("Interactive Robot Viewer", 1280, 720)
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([1, 1, 1, 1])
        self.scene.scene.show_axes(True)
        self.window.add_child(self.scene)

        em = self.window.theme.font_size
        margin = 0.5 * em
        self.panel = gui.Vert(0.25 * em, gui.Margins(margin))
        # === Checkbox 控制显示与否 ===
        self.show_robot = True
        self.show_spheres = True


        self.dist_label = gui.Label("Self-collision dist: 0.000")
        self.panel.add_child(self.dist_label)

        self.dist_nn_label = gui.Label("NN-predicted dist: 0.000")
        self.panel.add_child(self.dist_nn_label)


        self.checkbox_robot = gui.Checkbox("Show Robot")
        self.checkbox_robot.checked = True
        self.checkbox_robot.set_on_checked(lambda val: self._toggle_visibility("robot", val))
        self.panel.add_child(self.checkbox_robot)

        self.checkbox_spheres = gui.Checkbox("Show Spheres")
        self.checkbox_spheres.checked = True
        self.checkbox_spheres.set_on_checked(lambda val: self._toggle_visibility("spheres", val))
        self.panel.add_child(self.checkbox_spheres)


        self.sliders = []

        for i in range(7):
            s = gui.Slider(gui.Slider.DOUBLE)
            s.set_limits(-3.14, 3.14)
            s.double_value = self.q[i]
            s.set_on_value_changed(self.make_slider_callback(i))
            self.sliders.append(s)
            self.panel.add_child(gui.Label(f"Joint {i+1}"))
            self.panel.add_child(s)

        self.window.add_child(self.panel)

        random_button = gui.Button("Random Angle")
        random_button.set_on_clicked(self.set_random_joint_angles)
        self.panel.add_child(random_button)



        # === 重置按钮 ===
        reset_button = gui.Button("Reset")
        reset_button.set_on_clicked(self.reset_view_and_joints)
        self.panel.add_child(reset_button)


    
        self.window.set_on_layout(self._on_layout)
        self.camera_initialized = False
        self.update_scene()

    def _on_layout(self, layout_context):
        content_rect = self.window.content_rect
        panel_width = 300  # 滑块面板宽度
        self.scene.frame = gui.Rect(content_rect.x, content_rect.y,
                                    content_rect.width - panel_width,
                                    content_rect.height)
        self.panel.frame = gui.Rect(content_rect.get_right() - panel_width,
                                    content_rect.y,
                                    panel_width,
                                    content_rect.height)

    def _toggle_visibility(self, obj_type, visible):
        if obj_type == "robot":
            self.show_robot = visible
        elif obj_type == "spheres":
            self.show_spheres = visible
        self.update_scene()


    def reset_view_and_joints(self):
        # 恢复关节姿态
        self.q = self.q_init.copy()
        for i in range(len(self.sliders)):
            self.sliders[i].double_value = self.q[i]
        # 刷新场景
        self.camera_initialized = False
        self.update_scene()

    def set_random_joint_angles(self):
        self.q = np.random.uniform(low=-np.pi, high=np.pi, size=7)
        for i in range(len(self.sliders)):
            self.sliders[i].double_value = self.q[i]
        self.update_scene()


    def make_slider_callback(self, idx):
        def callback(val):
            self.q[idx] = val
            self.update_scene()
        return callback

    def update_scene(self):      
        self.scene.scene.clear_geometry()
        robot = get_robot_meshes(urdf_path, mesh_dir, self.q)
        spheres, dist_spheres, dist_nn = get_collision_spheres(self.q)
        self.dist_label.text = f"Self-collision dist (GT): {dist_spheres.item():.8f}"
        self.dist_nn_label.text = f"Self-collision dist (NN): {dist_nn.item():.8f}"

        if self.show_robot:
            for i, m in enumerate(robot):
                self.scene.scene.add_geometry(f"robot_{i}", m, self._material())

        if self.show_spheres:
            for i, m in enumerate(spheres):
                self.scene.scene.add_geometry(f"sphere_{i}", m, self._material([1, 0, 0]))

        if not hasattr(self, "camera_initialized") or not self.camera_initialized:
            bounds = self.scene.scene.bounding_box
            self.scene.setup_camera(60, bounds, bounds.get_center())
            self.camera_initialized = True

    def _material(self, color=[0.6, 0.6, 0.6]):
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        mat.base_color = color + [1.0]
        return mat
    

# === 主入口 ===
if __name__ == '__main__':
    gui.Application.instance.initialize()
    RobotGUI(np.array([0.0, 0.57, 0.0, 0.0, 0.57, 0.0, 0.0]))
    gui.Application.instance.run()

