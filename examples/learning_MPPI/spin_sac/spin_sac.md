## 对自动调整熵正则项 没有实现 

{
    "__pydevd_ret_val_dict":    {
        "EpochLogger.__init__": null
    },
    "ac_kwargs":        {
        "hidden_sizes": [
            256,
            256
        ]
    },
    "actor_critic":     "MLPActorCritic",
    "alpha":    0.2,
    "batch_size":       100,
    "epochs":   50,
    "exp_name": "sac",
    "gamma":    0.99,
    "logger":   {
        "<spinup.utils.logx.EpochLogger object at 0x7f94dccd7c40>":     {
            "epoch_dict":       {},
            "exp_name": "sac",
            "first_row":        true,
            "log_current_row":  {},
            "log_headers":      [],
            "output_dir":       "/home/zm/MotionPolicyNetworks/mpc_integral/spinningup/data/sac/sac_s0",
            "output_file":      {
                "<_io.TextIOWrapper name='/home/zm/MotionPolicyNetworks/mpc_integral/spinningup/data/sac/sac_s0/progress.txt' mode='w' encoding='UTF-8'>":      {
                    "mode":     "w"
                }
            }
        }
    },
    "logger_kwargs":    {
        "exp_name":     "sac",
        "output_dir":   "/home/zm/MotionPolicyNetworks/mpc_integral/spinningup/data/sac/sac_s0"
    },
    "lr":       0.001,
    "max_ep_len":       1000,
    "num_test_episodes":        10,
    "polyak":   0.995,
    "replay_size":      1000000,
    "save_freq":        1,
    "seed":     0,
    "start_steps":      10000,
    "steps_per_epoch":  4000,
    "update_after":     1000,
    "update_every":     50

Logging data to /home/zm/MotionPolicyNetworks/mpc_integral/spinningup/data/sac/sac_s0/progress.txt
‘‘‘

’’’

最终结果： caused by: ['/home/zm/Downloads/miniconda/home/zm/Downloads/miniconda/miniconda3/lib/python3.8/site-packages/tensorflow_io/python/ops/libtensorflow_io.so: undefined symbol: _ZTVN10tensorflow13GcsFileSystemE']
  warnings.warn(f"file system plugins are not loaded: {e}")
Warning: could not pickle state_dict.

未在尝试，
1. spin 均在cpu上计算实现
2. spin 更加规范: 对策略的更新要将Q_net 冻结； Q_net更新， Q_Target要冻结。 细节讨论的更加充分


