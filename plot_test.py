
from RLA.easy_plot.plot_func import plot_res_func
import os


prefix_dir = f'/Users/didi/project/mopo/log/policy_learn'
plot_res_func(prefix_dir, regs=['2021/01/30/18-46-*'],
              param_keys=["model_suffix"],
              value_keys=["training/sac_Q/q2"],
              resample=0,
              smooth_step=1)