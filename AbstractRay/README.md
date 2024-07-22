# AbstractRay
## Description
AbstractRay is a tool for evaluating machine learning models throught affine transformation using various backends and configurations. It supports custom models as well as popular pre-trained models such as VGG16, VGG19, ResNet, and SimpleCNN. The tool provides features for loading models, processing input images, generating zonotopes, and performing evaluations.

## Features

- Load custom models or pre-trained models (VGG16, VGG19, ResNet, SimpleCNN)
- Validate and transform input images
- Generate zonotopes for model evaluation
- Execute model evaluations using multiple workers and backends (CPU, CUDA)
- Visualize evaluation results

## Installation

1. Clone the repository:

    ```sh
    git clone https://gitlab/lan/guillaume/abstractray.git
    cd abstractray
    ```

2. Create a virtual environment and activate it:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Running the Model Evaluator

You can run the model evaluator using the command line interface (CLI) or through the Streamlit web interface.

### Command Line Interface (CLI)

1. **Evaluate a model using the CLI:**

    ```sh
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

### Streamlit Web Interface

In the fronted folder
1. **Run the Streamlit app:**


    ```sh
    streamlit run app.py
    ```
In the backend folder
    ```sh
    python main_stream.py
    ```
2. **Open your browser and navigate to:**

    ```
    http://localhost:8501
    ```

3. **Use the web interface to upload your model and input image, configure the evaluation settings, and run the evaluation.**

## Generating Documentation

1. **Install Sphinx and the Napoleon extension:**

    ```sh
    pip install sphinx sphinx-napoleon
    ```

2. **Initialize Sphinx in the `docs` directory:**

    ```sh
    cd docs
    sphinx-quickstart
    ```

3. **Configure `conf.py` to use the Napoleon extension:**

    ```python
    extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.napoleon',
    ]
    ```

4. **Generate `.rst` files for your modules:**

    ```sh
    sphinx-apidoc -o . ../my_project
    ```

5. **Build the HTML documentation:**

    ```sh
    make html
    ```

    The documentation will be generated in the `_build/html` directory.

