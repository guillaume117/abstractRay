#!/bin/bash

# Paramètres fixes
export PYTHONPATH=$PYTHONPATH:$(pwd)
NETWORK_FILE="./AbstratRay/src/CNN/simple_cnn_fashionmnist.pth"
INPUT_IMAGE="./test.jpeg"
NETWORK_NAME="simplecnn"
BACK_END="cpu"
RAM=0.5
RESIZE_INPUT=true
RESIZE_WIDTH=112
RESIZE_HEIGHT=112
BOX_INPUT=0
ADD_SYMBOL=1
RELEVANCE_DUMP=0
NOISE=0.0001
PARRALLEL_REL=1


for NUM_WORKER in {5..1}; do
    RESULT=$(python ./AbstractRay/main.py \
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
        --add_symbol $ADD_SYMBOL \
        --relevance_dump $RELEVANCE_DUMP\
        --parrallel_rel $PARRALLEL_REL)

    echo "Le résultat du script Python est : $RESULT"
done