import streamlit as st
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

st.title("Model Evaluator")
backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")


model_option = st.selectbox("Choose a model", ["Custom", "VGG16", "VGG19", "ResNet","SimpleCNN"])


if model_option == "Custom":
    uploaded_network = st.file_uploader("Upload your network model (.onnx or .pt/.pth)", type=["onnx", "pt", "pth"])
else:
    uploaded_network = None  

uploaded_image = st.file_uploader("Upload your input image", type=["jpg", "jpeg", "png"])

num_worker = st.number_input("Number of workers", min_value=0, value=1)
back_end = st.selectbox("Backend", ["cpu", "cuda"])
num_symbol = st.text_input("Number of symbols", value="Full")
noise = st.number_input("Noise level", value=0.00001)
RAM = st.number_input("Available RAM per worker", value=1.0)
add_symbol = st.selectbox("Add symbol from trash",[True,False])
relevance_dump = st.selectbox("Dump relevance policy",[False, True])
resize_input = st.checkbox("Resize input image", value=True)

if resize_input:
    resize_width = st.number_input("Resize width", min_value=1, value=224)
    resize_height = st.number_input("Resize height", min_value=1, value=224)
box_input = st.checkbox('Select a box for noising',value=False)
if box_input:
    box_x_min = st.number_input('X_min value for box')
    box_x_max = st.number_input('X_max value for box')
    box_y_min = st.number_input('Y_min value for box')
    box_y_max = st.number_input('Y_max value for box')

evaluation_status = st.empty()

def prepare_evaluation():
    if uploaded_image is not None:
        files = {
            'input_image': uploaded_image
        }
        if model_option == "Custom" and uploaded_network is not None:
            files['network'] = uploaded_network
            network_name  = "custom"
        else:
            network_name  = model_option.lower()

 
    

        data = {
            'network_name': network_name ,
            'num_worker': num_worker,
            'back_end': back_end,
            'box_input': box_input,
            'box_x_min': box_x_min if box_input else None,
            'box_x_max': box_x_max if box_input else None,
            'box_y_min': box_y_min if box_input else None,
            'box_y_max': box_y_max if box_input else None,
            'noise': noise,
            'RAM': RAM,
            'add_symbol': add_symbol,
            'relevance_dump': relevance_dump,
            'resize_input': resize_input,
            'resize_width': resize_width if resize_input else None,
            'resize_height': resize_height if resize_input else None
        }
        response = requests.post(f"{backend_url}/prepare_evaluation/", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            for message in result["messages"]:
                st.info(message)
            st.session_state.prepared = True
            st.session_state.messages = result["messages"]

   
            if resize_input:
                image = Image.open(uploaded_image)
                resized_image = image.resize((resize_width, resize_height))
                st.image(resized_image, caption='Resized Image')

        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            st.session_state.prepared = False
    else:
        st.error("Please upload the input image.")
        st.session_state.prepared = False

def execute_evaluation():
    response = requests.post(f"{backend_url}/execute_evaluation/")
    if response.status_code == 200:
        result = response.json()
        st.session_state.result = result
        st.session_state.relevance_index = 0
        plot_results(result)
        
       
       
            
        display_relevance_image(result, 0,noise)

    else:
        st.error(f"Error: {response.status_code} - {response.text}")

def plot_results(result):
    argmax = result["argmax"]
    true_values = [float(x) for x in result["true"]]
    center_values = [float(x) for x in result["center"]]
    min_values = [float(x) for x in result["min"]]
    max_values = [float(x) for x in result["max"]]

    x = np.arange(len(argmax))  

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - 0.2, true_values, 0.4, label='True', color='green')
    ax.bar(x, center_values, 0.4, label='Center', color='yellow')
    ax.errorbar(x, center_values, yerr=[np.array(center_values) - np.array(min_values), np.array(max_values) - np.array(center_values)], fmt='o', color='red', label='Min/Max')

    ax.set_xlabel('Class')
    ax.set_ylabel('Values')
    ax.set_title('Evaluation Results')
    ax.set_xticks(x)
    ax.set_xticklabels(argmax)
    ax.legend()

    st.pyplot(fig)

def display_relevance_image(result, index,noise):
    relevance_images = result['relevance']
    num_images = len(relevance_images)
    st.session_state.relevance_index = index % num_images

    relevance_image = np.array(relevance_images[st.session_state.relevance_index])/noise
    relevance_image = (relevance_image * 255).astype(np.uint8)
    print(max(relevance_image)-min(relevance_image))
    relevance_image = Image.fromarray(relevance_image).resize((resize_width, resize_height))

    original_image = Image.open(uploaded_image).resize((resize_width, resize_height)).convert("RGBA")
    relevance_image = relevance_image.convert("RGBA")
    

    blended_image = Image.blend(original_image, relevance_image, alpha=0.5)
    st.image(blended_image, caption=f'Relevance Image {st.session_state.relevance_index + 1}/{num_images}')

    if st.button("Previous", key="prev_button"):
        display_relevance_image(result, st.session_state.relevance_index - 1)
    if st.button("Next", key="next_button"):
        display_relevance_image(result, st.session_state.relevance_index + 1)

if 'prepared' not in st.session_state:
    st.session_state.prepared = False

if 'result' not in st.session_state:
    st.session_state.result = None

if st.button("Prepare Evaluation"):
    prepare_evaluation()

if st.session_state.prepared:
    st.success("Preparation completed. You can now execute the evaluation.")
    if st.button("Execute Evaluation"):
        execute_evaluation()
        if st.button("Interrupt Evaluation"):
            response = requests.post(f"{backend_url}/interrupt_evaluation/")
            if response.status_code == 200:
                st.warning("Evaluation interrupted by the user.")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")

if st.session_state.result:
    plot_results(st.session_state.result)


