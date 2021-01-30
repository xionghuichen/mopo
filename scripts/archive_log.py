"""
A script to archive benchmarking experiments. 

It is convenient to merge the archived experiments and the current task into tensorboard by:

tensorboard --logdir ./log/your_task/,./log/archived/

"""

from RLA.easy_log.log_tools import ArchiveLogTool
import argparse

def argsparser():
    parser = argparse.ArgumentParser("Archive Log")
    # reduce setting
    parser.add_argument('--sub_proj', type=str, default="")
    parser.add_argument('--task', type=str, default="policy_learn")
    parser.add_argument('--archive_name_as_task', type=str, default='archived')
    parser.add_argument('--reg', type=str)
    parser.add_argument('--remove', action='store_true')


    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = argsparser()
    dlt = ArchiveLogTool(proj_root='../', sub_proj=args.sub_proj, task=args.task, regex=args.reg,
                         archive_name_as_task=args.archive_name_as_task, remove=args.remove)
    dlt.archive_log()