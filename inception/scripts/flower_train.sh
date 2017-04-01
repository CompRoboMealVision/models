# Build the model. Note that we need to make sure the TensorFlow is ready to
# use before this as this command will not build TensorFlow.
bazel build inception/flowers_train

INCEPTION_MODEL_DIR="/data2/user_data/meal/inception-v3-model/inception-v3"

# Path to the downloaded Inception-v3 model.
MODEL_PATH="${INCEPTION_MODEL_DIR}/model.ckpt-157585"

# Directory where the flowers data resides.
FLOWERS_DATA_DIR=/data2/user_data/meal/flowers_data/

# Directory where to save the checkpoint and events files.
TRAIN_DIR=/data2/user_data/meal/flowers_train/

# Run the fine-tuning on the flowers data set starting from the pre-trained
# Imagenet-v3 model.

bazel-bin/inception/flowers_train \
  --train_dir="${TRAIN_DIR}" \
  --data_dir="${FLOWERS_DATA_DIR}" \
  --pretrained_model_checkpoint_path="${MODEL_PATH}" \
  --fine_tune=True \
  --initial_learning_rate=0.001 \
  --input_queue_memory_factor=1
