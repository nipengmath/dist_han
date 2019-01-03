#!/usr/bin/env bash
##########################################################
# where to write tfevents
OUTPUT_DIR="model-exports"
# experiment settings
TRAIN_BATCH=32
EVAL_BATCH=32
LR=0.001
EPOCHS=100
CPU_COUNT=10

# model para
PARA_MAX_NUM=7
PARA_MAX_LENGTH=18

# create a job name for the this run
prefix="example"
now=$(date +"%Y%m%d_%H_%M_%S")
JOB_NAME="$prefix"_"$now"
# locations locally or on the cloud for your files
TRAIN_FILES="data/forum/train.tfrecords"
EVAL_FILES="data/forum/dev.tfrecords"
TEST_FILES="data/forum/dev.tfrecords"
WORD_EMB_FILE="data/forum/word_emb.json"

##########################################################

GPU_ID=$1


# get current working directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# create folders if they don't exist of logs and outputs
mkdir -p $DIR/runlogs

# create a local job directory for checkpoints etc
JOB_DIR=${OUTPUT_DIR}/${JOB_NAME}


# start training
CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m initialisers.task \
        --job-dir ${JOB_DIR} \
        --train-batch-size ${TRAIN_BATCH} \
        --eval-batch-size ${EVAL_BATCH} \
        --cpu-count ${CPU_COUNT} \
        --learning-rate ${LR} \
        --num-epochs ${EPOCHS} \
        --train-files ${TRAIN_FILES} \
        --eval-files ${EVAL_FILES} \
        --test-files ${TEST_FILES} \
        --export-path "${OUTPUT_DIR}/exports"

echo "Job launched."
