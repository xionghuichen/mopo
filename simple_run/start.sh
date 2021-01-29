#!/bin/bash
. /home/luban/miniconda3/etc/profile.d/conda.sh
. /nfs/project/chenxionghui/.bashrc_maple
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/luban/.mujoco/mujoco200/bin
export MUJOCO_GL=osmesa
export DISPLAY=:0
conda activate codas
echo "---- env bashrc cli---"
python -V
conda info --env
echo "python main.py $*"
python main.py $*
sleep 2
echo "end shell"
wait