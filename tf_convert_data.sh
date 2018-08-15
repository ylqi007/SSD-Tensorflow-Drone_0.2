#!/usr/bin/env bash

DATASET_DIR=../data/VOC2007/train/
OUTPUT_DIR=./tfrecords
OUTPUT_NAME=voc_2007_train

python3 tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=${DATASET_DIR} \
    --output_name=${OUTPUT_NAME} \
    --output_dir=${OUTPUT_DIR}
