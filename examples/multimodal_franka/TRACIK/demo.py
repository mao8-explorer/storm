import numpy as np
from tracikpy import TracIKSolver
import time 

ee_pose = np.array([[[ 0.0525767 , -0.64690764, -0.7607537 , 0.        ],
                    [-0.90099786, -0.35923817,  0.24320937, 0.2       ],
                    [-0.43062577,  0.67265031, -0.60174996, 0.4       ],
                    [ 0.        ,  0.        ,  0.        , 1.        ]]])

list = ['Speed', 'Distance', 'Manipulation1', 'Manipulation2'] 
for i in range(len(list)):
    last = time.time()
    ik_solver = TracIKSolver(
            "content/assets/urdf/franka_description/franka_panda_no_gripper.urdf", 
            "panda_link0", 
            "ee_link", 
            timeout=0.05,
            solve_type=list[i]
            )
    
    qout = ik_solver.ik(ee_pose, qinit=np.zeros(ik_solver.number_of_joints))
    print(time.time() - last , qout)
    ee_out = ik_solver.fk(qout)
    ee_diff = np.linalg.inv(ee_pose) @ ee_out
    trans_err = np.linalg.norm(ee_diff[:3, 3], ord=1)
    angle_err = np.arccos(np.trace(ee_diff[:3, :3] - 1) / 2)

    assert trans_err < 1e-3
    assert angle_err < 1e-3 or angle_err - np.pi < 1e-3