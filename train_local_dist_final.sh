#!/usr/bin/env bash
##########################################################

JOB_NAME=$1
TASK_INDEX=$2
GPU=$3

# where to write tfevents
OUTPUT_DIR="model-exports"
MODEL_NAME="2018-12-06.v1"

# experiment settings
TRAIN_BATCH=512
EVAL_BATCH=512
TEST_BATCH=512
LR=0.001
EPOCHS=10
# create a job name for the this run
prefix="example"
now=$(date +"%Y%m%d_%H_%M_%S")

# locations locally or on the cloud for your files
TRAIN_FILES="data/train.tfrecords"
EVAL_FILES="data/val.tfrecords"
TEST_FILES="data/test.tfrecords"

# create a local job directory for checkpoints etc
JOB_DIR=${OUTPUT_DIR}/${MODEL_NAME}

##########################################################


if [[ -z $LD_LIBRARY_PATH || -z $CUDA_HOME  ]]; then
    echo ""
    echo "CUDA environment variables not set."
    echo "Consider adding them to your shell-rc."
    echo ""
    echo "Example:"
    echo "----------------------------------------------------------"
    echo 'LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"'
    echo 'LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64"'
    echo 'CUDA_HOME="/usr/local/cuda"'
    echo ""
fi


# needed to use virtualenvs
# set -euo pipefail

# get current working directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# create folders if they don't exist of logs and outputs
mkdir -p $DIR/runlogs


###################
# Add notes to the log file based on the current information about this training job close vim to start training
# useful if you are running lots of different experiments and you forget what values you used
echo "---"
echo "Learning Rate: ${LR}" >> training_log.md
echo "Epochs: ${EPOCHS}" >> training_log.md
echo "Batch Size (train/eval): ${TRAIN_BATCH}/ ${EVAL_BATCH}" >> training_log.md
echo "### Hypothesis
" >> training_log.md
echo "### Results
" >> training_log.md

###################

# Setup the distributed workflow. Ideally you would like at least twice as many workers as parameter servers, and
# each worker have a gpu associate with it, ps = parameter server

# This is an example for 3 GPUS, mocking the cloud training environment. The two workers use 2 GPUs and the master 1.
# Make sure specified ports are not being used
config="
{
    \"master\": [\"localhost:27182\"],
    \"ps\": [\"localhost:27183\"],
    \"worker\": [
        \"localhost:27184\",
        \"localhost:27185\"
        ]
}"

echo "Starting Training"

function run {
python3 -m initialisers.task \
        --job-dir ${JOB_DIR} \
        --job-name ${JOB_NAME} \
        --train-batch-size ${TRAIN_BATCH} \
        --eval-batch-size ${EVAL_BATCH} \
        --test-batch-size ${TEST_BATCH} \
        --learning-rate ${LR} \
        --num-epochs ${EPOCHS} \
        --train-files ${TRAIN_FILES} \
        --eval-files ${EVAL_FILES} \
        --test-files ${TEST_FILES} \
        --export-path "${OUTPUT_DIR}/exports" #\
              # &>runlogs/$1.log &
              # echo "$!" > runlogs/$1.pid
}


export CUDA_VISIBLE_DEVICES="$GPU"
# Parameter Server can be run on cpu
task="{\"type\": \"$JOB_NAME\", \"index\": $TASK_INDEX}"
export TF_CONFIG="{\"cluster\":${config}, \"task\":${task}}"
run
