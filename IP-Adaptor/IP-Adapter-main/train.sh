#!/bin/bash


BASE_MODEL_PATH="/stable-diffusion-v1-5"

IMAGE_ENCODER_PATH="/IP-Adapter/models/image_encoder"

DATASET_ROOT="/dataset_per_object"

OUTPUT_ROOT="checkpoints/IP-Adaptor-Final"


OBJECTS=( "bottle")

for OBJ in "${OBJECTS[@]}"; do

    JSON_FILE="${DATASET_ROOT}/${OBJ}/data.json"
    DATA_PATH="${DATASET_ROOT}/${OBJ}"
    OUTPUT_DIR="${OUTPUT_ROOT}/ip_adapter_${OBJ}"

    if [ ! -f "$JSON_FILE" ]; then
        echo "skip $JSON_FILE"
        continue
    fi


    CUDA_VISIBLE_DEVICES=3 accelerate launch  --gpu_ids='all' --num_processes 1 tutorial_train.py \
      --pretrained_model_name_or_path="$BASE_MODEL_PATH" \
      --image_encoder_path="$IMAGE_ENCODER_PATH" \
      --data_json_file="$JSON_FILE" \
      --data_root_path="$DATA_PATH" \
      --mixed_precision="fp16" \
      --resolution=512 \
      --train_batch_size=1 \
      --dataloader_num_workers=4 \
      --output_dir="$OUTPUT_DIR" \
      --learning_rate=1e-4 \
      --save_steps=500 \
      --num_train_epochs=200


    if [ $? -eq 0 ]; then
        echo "  [$(date '+%Y-%m-%d %H:%M:%S')] $OBJ   $OUTPUT_DIR"
    else
        echo "  [$(date '+%Y-%m-%d %H:%M:%S')] $OBJ  "

    fi

    sleep 5

done

