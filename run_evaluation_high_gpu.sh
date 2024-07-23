export PYTHONPATH=$PYTHONPATH:$(pwd)
NETWORK_FILE="./AbstratRay/src/CNN/simple_cnn_fashionmnist.pth"
INPUT_IMAGE="./cut_dog.jpeg"
NETWORK_NAME="vgg16"
NUM_WORKER=0
BACK_END="cuda"
RAM=48
RESIZE_INPUT="True"
RESIZE_WIDTH=224    
RESISE_HEIGHT=224
BOX_INPUT="False"
ADD_SYMBOL="True"
RELEVANCE_DUMP="False"


NOISE=0.00001

while (( $(echo "$NOISE <= 0.0001" | bc -l) )); do
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
        --relevance_dump $RELEVANCE_DUMP\
     NOISE=$(echo "$NOISE + 0.00001" | bc)

done