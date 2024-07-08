import streamlit as st
import requests
import json
import matplotlib.pyplot as plt
import numpy as np

st.title("Model Evaluator")

uploaded_network = st.file_uploader("Upload your network model (.onnx or .pt/.pth)", type=["onnx", "pt", "pth"])
uploaded_image = st.file_uploader("Upload your input image", type=["jpg", "jpeg", "png"])

num_worker = st.number_input("Number of workers", min_value=0, value=1)
back_end = st.selectbox("Backend", ["cpu", "cuda"])
num_symbol = st.text_input("Number of symbols", value="Full")
noise = st.number_input("Noise level", value=0.00001)
RAM = st.number_input("Available RAM per worker", value=1.0)

# Placeholder for evaluation status
evaluation_status = st.empty()

def prepare_evaluation():
    if uploaded_network is not None and uploaded_image is not None:
        files = {
            'network': uploaded_network,
            'input_image': uploaded_image
        }
        # Convert num_symbol to str if it's an integer
        try:
            num_symbol_value = int(num_symbol)
        except ValueError:
            num_symbol_value = str(num_symbol)

        data = {
            'num_worker': num_worker,
            'back_end': back_end,
            'num_symbol': num_symbol_value,
            'noise': noise,
            'RAM': RAM
        }
        response = requests.post("http://localhost:8000/prepare_evaluation/", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            for message in result["messages"]:
                st.info(message)
            st.session_state.prepared = True
            st.session_state.messages = result["messages"]
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            st.session_state.prepared = False
    else:
        st.error("Please upload both the network model and the input image.")
        st.session_state.prepared = False

def execute_evaluation():
    response = requests.post("http://localhost:8000/execute_evaluation/")
    if response.status_code == 200:
        result = response.json()
        st.session_state.result = result
        plot_results(result)
    else:
        st.error(f"Error: {response.status_code} - {response.text}")

def plot_results(result):
    print(result)
    argmax = result["argmax"][0]
    true_values = [float(x) for x in result["true"][0]]
    center_values = [float(x) for x in result["center"][0]]
    min_values = [float(x) for x in result["min"][0]]
    max_values = [float(x) for x in result["max"][0]]

    x = np.arange(len(argmax))  # Les indices des classes

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

if st.session_state.result:
    plot_results(st.session_state.result)

if st.button("Interrupt Evaluation"):
    response = requests.post("http://localhost:8000/interrupt_evaluation/")
    if response.status_code == 200:
        st.warning("Evaluation interrupted by the user.")
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
