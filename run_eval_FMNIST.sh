#!/bin/bash

# Paramètres fixes
export PYTHONPATH=$PYTHONPATH:$(pwd)
NETWORK_FILE="./AbstratRay/src/CNN/simple_cnn_fashionmnist.pth"
INPUT_IMAGE="./test.jpeg"
NETWORK_NAME="simplecnn"
BACK_END="cpu"
RAM=12
RESIZE_INPUT=0
RESIZE_WIDTH=224        
RESIZE_HEIGHT=224
BOX_INPUT=0
MIN=82
MAX=142
ADD_SYMBOL=1
RELEVANCE_DUMP=0
NOISE=0.01
PARRALLEL_REL=0


for NUM_WORKER in {0..0}; do
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
        --resize_height $RESIZE_HEIGHT\
        --box_input $BOX_INPUT \
        --box_x_min $MIN\
        --box_x_max $MAX\
        --box_y_min $MIN\
        --box_y_max $MAX\
        --add_symbol $ADD_SYMBOL \
        --relevance_dump $RELEVANCE_DUMP\
        --parrallel_rel $PARRALLEL_REL\

 
done