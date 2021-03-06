#!/bin/bash

# Where the pre-trained Inception Resnet V2 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/root/Desktop/LogoReader/data/slim_test/checkpoints

# Where the pre-trained Inception Resnet V2 checkpoint is saved to.
MODEL_NAME=inception_resnet_v2

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/root/Desktop/LogoReader/data/slim_test/training/${MODEL_NAME}

# Where the dataset is saved to.
DATASET_DIR=/root/Downloads/CarLogos51

echo ----------------Begin to train----------------
python my_train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=carlogos51 \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/${MODEL_NAME}.ckpt \
  --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
  --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits,InceptionResnetV2/Conv2d_7b_1x1,InceptionResnetV2/Block8 \
  --max_number_of_steps=6000 \
  --batch_size=100 \
  --learning_rate=0.01 \
  --learning_rate_decay_type=exponential \
  --num_epochs_per_decay=50 \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=adagrad \
  --weight_decay=0.00004 \
  --preprocessing_name='carlogo'

echo ------------------Final evaluation----------------
python my_eval_image_classifier.py \
  --alsologtostderr \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=carlogos51 \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --preprocessing_name='carlogo'

