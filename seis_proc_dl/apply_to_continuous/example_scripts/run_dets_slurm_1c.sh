#!/usr/bin/env bash
#SBATCH --time=17:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=4G
#SBATCH --partition=notchpeak-shared
#SBATCH --qos=notchpeak
#SBATCH --account=koper
#SBATCH -C "csl|skl|rom" # MAKE SURE TO SET THIS IF USING NOTCHPEAK
#SBATCH -o ./logs_1C/slurm-%A-%a.out-%N # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e ./logs_1C/slurm-%A-%a.err-%N # name of the stderr, using job and first node values
#SBATCH --array=1-20

# Purpose: To run phase detectors on continous data for various stations.
# Author: Alysha Armstrong distributed under the MIT license.
# Usage: sbatch

mkdir -p "./logs_1C"

year=2023
month=2
day=1
n_days=334
n_comps=1
CONFIG_FILE="./detector_config.py"
STAT_FILE="../station_lists/station.list.db.1C.2023-01-01.2024-01-01.txt"
RUN_SCRIPT_PATH=~/PycharmProjects/seis_proc_dl/seis_proc_dl/apply_to_continuous

# Set some directory information
PYTHON_DIR=~/software/pkg/miniforge3/envs/atc/bin

# Read in the number of stations in the file
read NSTATS < <(sed -n 1p $STAT_FILE)
# echo "Number of stations in file: ${NSTATS}"

# Check that the number of staions matches the size of the slurm array
if [ $NSTATS != $SLURM_ARRAY_TASK_COUNT ]; then
        echo "Number of slurm arrays does not match the number of stations in the station list. Exiting."
        exit 1
fi

# Read in the station and channel prefix (e.g., HH) from file
# The first line of the file has the number of stations
read NET STAT LOC CHAN < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $STAT_FILE)
echo $NET $STAT $LOC $CHAN 

# Execute code
${PYTHON_DIR}/python -u ${RUN_SCRIPT_PATH}/run_apply_detector.py --net $NET -s $STAT -l $LOC -c $CHAN  -y $year -m $month -d $day -n $n_days --cfg $CONFIG_FILE --ncomps $n_comps
