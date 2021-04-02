#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N MatchPDE
#$ -cwd
#$ -l h_rt=10:00:00
#$ -l h_vmem=8G
#$ -M s1415551@ed.ac.uk
#$ -m be
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 600 minutes: -l h_rt
#  memory limit of 1 Gbyte: -l h_vmem
# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load Matlab
module load anaconda

# Run the program
source activate IPS
python OneClusterVaryGamma.py
