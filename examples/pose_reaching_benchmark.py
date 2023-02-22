
import isaacgym 
from omegaconf import DictConfig, OmegaConf

import os
import hydra




@hydra.main(config_name="config", config_path="../content/configs/gym")
def main(cfg: DictConfig):
    import isaacgymenvs
    from storm_kit.gym import tasks

    envs = isaacgymenvs.make(
        cfg.seed, 
        cfg.task_name, 
        cfg.task.env.numEnvs, 
        cfg.sim_device,
        cfg.rl_device,
        cfg.graphics_device_id,
        cfg.headless,
        cfg.multi_gpu,
        cfg.capture_video,
        cfg.force_render,
        cfg,
        # **kwargs,
    )
    while True:
        envs.step()
        envs.render()
    # input('....')

if __name__ == "__main__":
    main()