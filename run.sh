#!/bin/bash

BERT_BASE_DIR=$PWD/../py-eflect/experiments/resources/bert
GLUE_DIR=$PWD/../py-eflect/experiments/third_party/GLUE-baselines/glue_data

SIZE=tiny
TASK=CoLA

rm -rf /tmp/output/*

echo ${PWD}/data

python3.8 run_classifier.py \
  --task_name=${TASK} \
  --do_train=true \
  --do_eval=true \
  --data_dir=${GLUE_DIR}/${TASK} \
  --vocab_file=${BERT_BASE_DIR}/${SIZE}/vocab.txt \
  --bert_config_file=${BERT_BASE_DIR}/${SIZE}/bert_config.json \
  --init_checkpoint=${BERT_BASE_DIR}/${SIZE}/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=/tmp/output \
  --tracing_output_dir=${PWD}/data
