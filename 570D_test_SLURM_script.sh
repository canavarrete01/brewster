#!/bin/bash
#SBATCH -J G570D_check_108
#SBATCH --output=G570_check_108.o%j
#SBATCH --error=G570_check_108.e%j
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH --time=30:00:00
#SBATCH --mail-user=cnavarrete@amnh.org
#SBATCH --mail-type=ALL

source ~/.bashrc

declare -xr WDIR="/home/cnavarrete/mendel-nas1/BDNYC/brewster/"

declare PATH=${PATH}:${WDIR}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${WDIR}:~/

# Active python 3 environment
source activate retrieval

module load OpenMPI/openmpi-4.0.4

time_start=`date '+%T%t%d_%h_06'`
  
echo ------------------------------------------------------
echo -n 'Job is running on node '; echo  $SLURM_JOB_NODELIST
echo ------------------------------------------------------
echo SBATCH: sbatch is running on $SLURM_SUBMIT_HOST
echo SBATCH: originating queue is $SLURM_SUBMIT_PARTITION
echo SBATCH: executing queue is $SLURM_JOB_PARTITION
echo SBATCH: working directory is $SLURM_SUBMIT_DIR
echo SBATCH: job identifier is $SLURM_JOBID
echo SBATCH: job name is $SLURM_JOB_NAME
echo SBATCH: node file is $SLURM_JOB_NODELIST
echo SBATCH: current home directory is $SLURM_SUBMIT_HOME
echo SBATCH: PATH = $SLURM_SUBMIT_PATH
echo ------------------------------------------------------


cd ${WDIR}


mpirun python brewster_template.py > /home/cnavarrete/mendel-nas1/BDNYC/brewster/Results/G570D_check.log

time_end=`date '+%T%t%d_%h_06'`
echo Started at: $time_start
echo Ended at: $time_end
echo ------------------------------------------------------
echo Job ends

