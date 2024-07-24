.. AbstractRay documentation master file, created by
   sphinx-quickstart on Sun Jul 21 20:11:46 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to AbstractRay's documentation!
=======================================

AbstractRay is a tool for evaluating machine learning models throught affine transformation using various backends and configurations. It supports custom models as well as popular pre-trained models such as VGG16, VGG19, ResNet, and SimpleCNN. The tool provides features for loading models, processing input images, generating zonotopes, and performing evaluations.

Features
========

- Load custom models (ONNX) or pre-trained models (VGG16, VGG19, ResNet, SimpleCNN)
- Validate and transform input  images
- Generate zonotopes for model evaluation
- Execute model evaluations using multiple workers on Ray Kubernetes Cluster and backends (CPU, CUDA)
- Visualize evaluation results

 Command Line Interface (CLI)
=============================
You can run the model evaluator using the command line interface (CLI) or through the Streamlit web interface.
1. **Evaluate a model using the CLI:**

``` sh
python evaluate_model.py --network_file path/to/model.onnx --input_image path/to/image.jpg --network_name custom --num_worker 4 --back_end cpu --noise 0.00001 --RAM 4.0 --resize_input True --resize_width 224 --resize_height 224 --box_input False --add_symbol True --relevance_dump True
```

2. **Arguments:**

    - `--network_file`: Path to the network file (.onnx, .pt, .pth) (required for custom models)
    - `--input_image`: Path to the input image (required)
    - `--network_name`: Name of the network (`custom`, `vgg16`, `vgg19`, `resnet`, `simplecnn`) (required)
    - `--num_worker`: Number of workers (required)
    - `--back_end`: Backend device (`cpu`, `cuda`) (required)
    - `--noise`: Noise level (required)
    - `--RAM`: Available RAM (required)
    - `--resize_input`: Whether to resize the input image (required)
    - `--resize_width`: Width to resize the input image to (optional)
    - `--resize_height`: Height to resize the input image to (optional)
    - `--box_input`: Whether to use a bounding box for the input (required)
    - `--box_x_min`: Minimum x value for the bounding box (optional)
    - `--box_x_max`: Maximum x value for the bounding box (optional)
    - `--box_y_min`: Minimum y value for the bounding box (optional)
    - `--box_y_max`: Maximum y value for the bounding box (optional)
    - `--add_symbol`: Whether to add symbols (required)
    - `--relevance_dump`: Whether to dump relevance information (required)
    - `--model_last_layer` : For evaluate only the first args layers (optional) 

.. toctree::
   :maxdepth: 2
   :caption: Contents:

  
   AbstractRay


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


