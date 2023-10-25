import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

class WorldMoveableImageCollision:
    def __init__(self, 
                 world_image="/home/zm/MotionPolicyNetworks/storm_ws/storm/content/assets/collision_maps/collision_map_cem.png", 
                 tensor_args={'device':"cpu", 'dtype':torch.float32}):
        self.scene_im = None
        self.tensor_args = tensor_args
        im = cv2.imread(world_image, 0)
        _, im = cv2.threshold(im, 10, 255, cv2.THRESH_BINARY)
        self.im = im
        shift = 3
        self.movelist = np.float32([
            [[1, 0, -shift], [0, 1, 0]],
            [[1, 0, shift], [0, 1, 0]]])
        self.step_move = 20
        self.move_ind = 10
        plt.figure()
        self.ax = plt.subplot(1, 1, 1)
        self.ax.invert_yaxis()  # 翻转y轴

    def run(self):  
        """
        SDF  potential and gradient update
        """
        # load image and move_it
        rows, cols = self.im.shape
        ind = self.move_ind % (2 * self.step_move) // self.step_move
        self.move_ind += 1
        M_left = self.movelist[ind]
        self.im = cv2.warpAffine(self.im, M_left, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=255)

        im_obstacle = cv2.bitwise_not(self.im)
        dist_obstacle = cv2.distanceTransform(im_obstacle, cv2.DIST_L2, 3)
        dist_outside = cv2.distanceTransform(self.im, cv2.DIST_L2, 3)

        dist_map = dist_obstacle - dist_outside
        self.dist_map = torch.as_tensor(dist_map, **self.tensor_args)

        # 在将2D图像转成SDF后，现在需要刻画SDF的梯度并尝试可视化梯度效果
        # 大致的运算：
        grad_y, grad_x = torch.gradient(self.dist_map)
        # 可视化 grad_x ， grad_y 应该如何设计？ 
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-10)
        # 这里的转向问题是坑，要多调试多看看 
        norm_grad_x, norm_grad_y = -grad_x / gradient_magnitude, -grad_y / gradient_magnitude
        # 定义箭头位置
        y_range, x_range = self.dist_map.shape
        x_step = y_step =  40
        x = np.arange(0, x_range, x_step)
        y = np.arange(0, y_range, y_step)
        X, Y = np.meshgrid(x, y)
        # 绘制箭头
        self.ax.cla()
        self.ax.set_xlim(0, x_range)
        self.ax.set_ylim(0, y_range)
        self.ax.imshow(self.dist_map.cpu())
        self.ax.quiver(X, Y, 
                       norm_grad_x[::x_step, ::y_step].cpu().numpy(), 
                       norm_grad_y[::x_step, ::y_step].cpu().numpy(), 
                       gradient_magnitude[::x_step, ::y_step].cpu().numpy(), 
                       cmap=plt.cm.jet)
        plt.pause(1e-10)

updatesdf = WorldMoveableImageCollision()
while True:
    updatesdf.run()

