#!/bin/bash

# Paramètres fixes
export PYTHONPATH=$PYTHONPATH:$(pwd)
NETWORK_FILE="./AbstratRay/src/CNN/simple_cnn_fashionmnist.pth"
INPUT_IMAGE="./test.jpeg"
NETWORK_NAME="simplecnn"
BACK_END="cpu"
RAM=64 
RESIZE_INPUT="True"
RESIZE_WIDTH=112    
RESISE_HEIGHT=112
BOX_INPUT="False"
ADD_SYMBOL="True"
RELEVANCE_DUMP="False"

NOISE=0.00001

for NUM_WORKER in {2..1}; do
    python ./AbstractRay/main.py \
        --network_file $NETWORK_FILE \
        --input_image $INPUT_IMAGE \
        --network_name $NETWORK_NAME \
        --num_worker $NUM_WORKER \
        --back_end $BACK_END \
        --noise $NOISE \
        --RAM $RAM \
        --resize_input $RESIZE_INPUT \
        --resize_width $RESIZE_WIDTH \
        --resize_height $RESISE_HEIGHT\
        --box_input $BOX_INPUT \
        --add_symbol $ADD_SYMBOL \
        --relevance_dump $RELEVANCE_DUMP;
done