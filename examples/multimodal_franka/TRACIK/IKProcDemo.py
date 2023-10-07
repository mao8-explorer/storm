
import multiprocessing as mp
import queue
from tracikpy import TracIKSolver
import numpy as np
import time

# 1. 若使用collections.deque 尽管能实现类似容量限制 先进先出等，但是进程间无法共享内存
# 2. 若使用queue，尽管可以借助multiprocessing实现进程间内存共享，但是只设置maxsize的话，超过限制，默认阻塞，等待队列处理完数据给空。
# 使用基于queue自定义类的方式解决这一问题
# 3. 使用get_nowait() 避免阻塞 一定不要阻塞
class LimitedQueue:
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.queue = mp.Queue(maxsize=maxsize)

    def put(self, item):
        if self.queue.qsize() >= self.maxsize:
            try:
                self.queue.get_nowait()  # 从队列中获取旧数据以腾出空间
            except queue.Empty :
                pass
        self.queue.put(item)

    def get(self):
        return self.queue.get_nowait()


class IKProc(mp.Process):
    """
    Used for finding ik in parallel.
    """

    def __init__(
        self,
        output_queue,
        input_queue_maxsize = 5,
    ):
        """
        Args:
        output_queue: mp.Queue, the queue that all the output data
            that is computed is added to.
        """
        super().__init__()
        self.solve_types = ['Speed', 'Distance', 'Manipulation1', 'Manipulation2'] 
        self.output_queue = output_queue
        self.input_queue = LimitedQueue(input_queue_maxsize)
        self.ik_solver = TracIKSolver(
            "content/assets/urdf/franka_description/franka_panda_no_gripper.urdf",
            "panda_link0",
            "ee_link",
            timeout=0.05,
            solve_type= self.solve_types[2],
        )

    def _ik(self, ee_pose, qinit):
        qout = self.ik_solver.ik(
            ee_pose, qinit[: self.ik_solver.number_of_joints]
        )
        return qout

    def run(self):
        """
        the main function of each path collector process.
        """
        while True:
            try:
                request = self.input_queue.get()
                
            except queue.Empty :
                "配合 self.input_queue.get(timeout=1) 当超过1s 会报_queue.Empty ,使用try ... except ...捕获"
                continue
            if request[0] == "ik":
                # print("ik请求填写...")
                self.output_queue.put(
                    (request[3], self._ik(request[1], request[2]))
                )

    def ik(self, grasp, init_q, ind=None):
        self.input_queue.put(("ik", grasp, init_q, ind))
        # print(self.input_queue.queue.qsize())


class MPPI:
    def __init__(self , num_proc = 1):
        
        self.num_proc = num_proc
        self.maxsize = 5
        self.output_queue = LimitedQueue(self.maxsize)
        self.ik_procs = []
        for i in range(num_proc):
            self.ik_procs.append(
                IKProc(
                    self.output_queue,
                    input_queue_maxsize=self.maxsize,
                )
            )
            self.ik_procs[-1].daemon = True #守护进程 主进程结束 IKProc进程随之结束
            self.ik_procs[-1].start()
    
    def test(self, final_poses, qinit):
        # Check ik for all final grasps distributed between processes
        final_ik = []
        # 模拟franka_reacher_MPC设计
        for i, p in enumerate(final_poses):
            # self.ik_procs[i % self.num_proc].ik(p, qinit, ind = i)
            self.ik_procs[-1].ik(p, qinit, ind = i)
            time.sleep(0.04) # like mpc_policy run
            try :
                # print("output_queue请求中 ... ...")
                output = self.output_queue.get()
                print("看看现在是第几个: ",output[0])
                final_ik.append(output)
            except queue.Empty:
                print("别急 还没好-----")
                "针对 output_queue队列为空的问题 会出现queue.Empty的情况发生"
                continue
        # If no IK or IK is wildly different
        print(final_ik)
        

def main():
    ee_pose = np.array([[[ 0.0525767 , -0.64690764, -0.7607537 , 0.        ],
                        [-0.90099786, -0.35923817,  0.24320937, 0.2       ],
                        [-0.43062577,  0.67265031, -0.60174996, 0.4       ],
                        [ 0.        ,  0.        ,  0.        , 1.        ]]])

    # 使用tile函数重复ee_pose矩阵5次
    repeated_ee_pose = np.tile(ee_pose, (100, 1, 1))
    qinit = np.zeros(7)
    policy = MPPI()
    policy.test(repeated_ee_pose, qinit)
    time.sleep(1)


if __name__ == "__main__":
    main()