#!/bin/bash

# Paramètres fixes
export PYTHONPATH=$PYTHONPATH:"./app"
NETWORK_FILE=".app/src/CNN/simple_cnn_fashionmnist.pth"
INPUT_IMAGE="./test.jpeg"
NETWORK_NAME="simplecnn"
NUM_WORKER=0
BACK_END="cpu"
RAM=8.0
RESIZE_INPUT="False"
BOX_INPUT="False"
ADD_SYMBOL="True"
RELEVANCE_DUMP="False"


NOISE=0.0005

while (( $(echo "$NOISE <= 0.001" | bc -l) )); do
    python app/main.py \
        --network_file $NETWORK_FILE \
        --input_image $INPUT_IMAGE \
        --network_name $NETWORK_NAME \
        --num_worker $NUM_WORKER \
        --back_end $BACK_END \
        --noise $NOISE \
        --RAM $RAM \
        --resize_input $RESIZE_INPUT \
        --box_input $BOX_INPUT \
        --add_symbol $ADD_SYMBOL \
        --relevance_dump $RELEVANCE_DUMP
     NOISE=$(echo "$NOISE + 0.0005" | bc)
done
