#!/usr/bin/env bash
##########################################################

JOB_NAME=$1
TASK_INDEX=$2
GPU=$3

# where to write tfevents
OUTPUT_DIR="model-exports"
MODEL_NAME="2018-12-28.v1"

# experiment settings
TRAIN_BATCH=512
EVAL_BATCH=512
TEST_BATCH=512
LR=0.001
EPOCHS=10
CPU_COUNT=10

# model para
PARA_MAX_NUM=7
PARA_MAX_LENGTH=18

# create a job name for the this run
prefix="example"
now=$(date +"%Y%m%d_%H_%M_%S")

# locations locally or on the cloud for your files
TRAIN_FILES="data/forum/train.tfrecords"
EVAL_FILES="data/forum/dev.tfrecords"
TEST_FILES="data/forum/dev.tfrecords"
WORD_EMB_FILE="data/forum/word_emb.json"


# create a local job directory for checkpoints etc
JOB_DIR=${OUTPUT_DIR}/${MODEL_NAME}

##########################################################

# get current working directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# create folders if they don't exist of logs and outputs
mkdir -p $DIR/runlogs


config="
{
    \"master\": [\"192.168.192.166:27182\"],
    \"ps\": [\"192.168.192.166:27183\"],
    \"worker\": [
        \"192.168.192.166:27184\",
        \"192.168.192.166:27185\"
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
