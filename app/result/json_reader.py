import os
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def process_model_data(json_data):
    # Parsing the JSON data
    layers = json_data["layers"]
    context = json_data["context"]
    add_symbol = context.get("add_symbol", False)
    
    context_info = (
        f"Model Name: {context.get('model_name', 'N/A')}\n"
        f"Input tensor shape:{context.get('input_tensor_size','N/A')}\n"
        f"Workers: {context.get('num_workers', 0)}\n"
        f"RAM: {context.get('available_RAM', 0)} GB\n"
        f"Device: {context.get('device', 'N/A')}\n"
        f"Add Symbol: {context.get('add_symbol', 'N/A')}\n"
        f"Renew Abstract Domain: {context.get('renew_abstract_domain', 'N/A')}\n"
        f"Verbose: {context.get('verbose', 'N/A')}\n"
        f"Noise Level: {context.get('noise_level', 'N/A')}\n"
        f"Num Symbols: {context.get('num_symbols', 'N/A')}\n"
        f"Process ended:{context.get('process_ended','N/A')}"
    )
    
    # Initialize lists to store cumulative times, memory gains, and num_symbols if applicable
    cumulative_times = []
    memory_gains = []
    layer_names = []
    num_symbols = [] if add_symbol else None
    
    cumulative_time = 0
    for layer in layers:
        layer_names.append(layer["layer_name"])
        computation_time = layer["computation_time"]
        memory_gain_percentage = layer["memory_gain_percentage"]
        cumulative_time += computation_time
        cumulative_times.append(cumulative_time)
        memory_gains.append(memory_gain_percentage)
        
        if add_symbol:
            num_symbols.append(layer["num_symbols"])
    
    # Plotting with two different y-axes
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    color = 'tab:blue'
    ax1.set_xlabel('Layer Name')
    ax1.set_ylabel('Memory Gain Percentage', color=color)
    ax1.plot(layer_names, memory_gains, marker='o', color=color, label='Memory Gain Sparse vs Dense Percentage')
    ax1.tick_params(axis='y', labelcolor=color)
    plt.xticks(rotation=90)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Cumulative Computation Time (s)', color=color)
    ax2.plot(layer_names, cumulative_times, marker='x', color=color, label='Cumulative Computation Time')
    ax2.tick_params(axis='y', labelcolor=color)
    
    if add_symbol:
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))  # Move the third y-axis outward
        color = 'tab:green'
        ax3.set_ylabel('Number of Symbols', color=color)
        ax3.plot(layer_names, num_symbols, marker='s', color=color, label='Number of Symbols')
        ax3.tick_params(axis='y', labelcolor=color)
    
    plt.title('Memory Gain Percentage and Cumulative Computation Time by Layer')
    fig.tight_layout()
    
    # Improve readability of x-axis labels
    
    plt.grid(True)
    
    # Add context information as a legend
    plt.legend(loc='upper left', bbox_to_anchor=(0.03, 0.85), title="Context", labels=[context_info])
    
    return fig

def load_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def generate_graphs_from_folder(folder_path, output_pdf):
    pdf_pages = PdfPages(output_pdf)
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            print('resr')
            file_path = os.path.join(folder_path, filename)
            json_data = load_json(file_path)
            fig = process_model_data(json_data)
            pdf_pages.savefig(fig)
            plt.close(fig)
    
    pdf_pages.close()

# Folder path containing JSON files
folder_path = './'
# Output PDF file
output_pdf = './test.pdf'

# Generate the graphs and save them to a PDF
generate_graphs_from_folder(folder_path, output_pdf)
