"""
A script to delete useless experimental logs by regex string.

"""
from RLA.easy_log.log_tools import DeleteLogTool
import argparse

def argsparser():
    parser = argparse.ArgumentParser("Delete Log")
    # reduce setting
    parser.add_argument('--sub_proj', type=str, default="")
    parser.add_argument('--task', type=str, default="policy_learn")
    parser.add_argument('--reg', type=str)


    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = argsparser()
    dlt = DeleteLogTool(proj_root='../', sub_proj=args.sub_proj, task=args.task, regex=args.reg)
    dlt.delete_related_log()