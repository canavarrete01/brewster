#PBS -S /bin/tcsh
#PBS -N bb_5gas_test
#PBS -m abe
#PBS -l select=1:ncpus=24:mpiprocs=24:model=has
#PBS -l walltime=12:00:00
#PBS -k oe
#PBS -r n
#PBS -q long
#PBS -W group_list=s1152
source /usr/share/modules/init/csh
module load mpi-intel/4.1.1.036 comp-intel/2015.0.090 python/2.7.10
source /usr/local/lib/global.cshrc


setenv PATH ${PATH}:/home1/bburning/retrievals/marks_RT_version:/u/scicon/tools/bin
setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/home1/bburning/retrievals/marks_RT_version
#setenv MPI_BUFS_PER_PROC 512
#setenv OMP_NUM_THREADS 20




set time_start=`date '+%T%t%d_%h_06'`
  
echo ------------------------------------------------------
echo -n 'Job is running on node '; cat $PBS_NODEFILE
echo ------------------------------------------------------
echo PBS: qsub is running on $PBS_O_HOST
echo PBS: originating queue is $PBS_O_QUEUE
echo PBS: executing queue is $PBS_QUEUE
echo PBS: working directory is $PBS_O_WORKDIR
echo PBS: execution mode is $PBS_ENVIRONMENT
echo PBS: job identifier is $PBS_JOBID
echo PBS: job name is $PBS_JOBNAME
echo PBS: node file is $PBS_NODEFILE
echo PBS: current home directory is $PBS_O_HOME
echo PBS: PATH = $PBS_O_PATH
echo ------------------------------------------------------



cd /home1/bburning/retrievals/marks_RT_version

mpdboot --file=$PBS_NODEFILE --ncpus=1 --totalnum=`cat $PBS_NODEFILE | sort -u | wc -l` --ifhn=`head -1 $PBS_NODEFILE` --rsh=ssh --mpd=`which mpd` --ordered

mpiexec -machinefile $PBS_NODEFILE -np 24 python brewster_5g.py > brew_5gastest.log

set time_end=`date '+%T%t%d_%h_06'`
echo Started at: $time_start
echo Ended at: $time_end
echo ------------------------------------------------------
echo Job ends

# terminate the MPD daemon

mpdallexit
