#!/bin/bash
chmod a+x  /home/luban/.bashrc
. /home/luban/.bashrc
conda activate codas
echo "---- env bashrc cli---"
python -V
conda info --env
echo "python main.py $*"
python main.py $*
sleep 2
echo "end shell"
wait