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
cd ..
cd RLAssistant
pip install -e .
echo "--- complete install RLA ---"
cd ..
cd mopo
pip install -e .
echo "--- complete install MOPO ---"
cd simple_run
ls -lha /home/luban/.d4rl/datasets/

echo "python main.py $* --seed 8"
{
  python main.py $* --seed 8
}&
echo "python main.py $* --seed 88"
sleep 2
{
  python main.py $* --seed 88
}&
sleep 2
echo "python main.py $* --seed 888"
{
  python main.py $* --seed 888
}&
sleep 2
echo "end shell"
wait