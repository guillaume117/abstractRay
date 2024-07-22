import os
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def process_model_data(json_data):
    """
    Process the model data from the JSON and generate the corresponding plots.

    Args:
        json_data (dict): The JSON data containing model information and layer statistics.

    Returns:
        tuple: A tuple containing the generated figure and context information.
    """
    layers = json_data["layers"]
    context = json_data["context"]
    add_symbol = context.get("add_symbol", False)
    
    context_info = (
        f"Model Name: {context.get('model_name', 'N/A')}\n"
        f"Input tensor shape: {context.get('input_tensor_size', 'N/A')}\n"
        f"Workers: {context.get('num_workers', 0)}\n"
        f"RAM: {context.get('available_RAM', 0)} GB\n"
        f"Device: {context.get('device', 'N/A')}\n"
        f"Add Symbol: {context.get('add_symbol', 'N/A')}\n"
        f"Renew Abstract Domain: {context.get('renew_abstract_domain', 'N/A')}\n"
        f"Verbose: {context.get('verbose', 'N/A')}\n"
        f"Noise Level: {context.get('noise_level', 'N/A')}\n"
        f"Num Symbols: {context.get('num_symbols', 'N/A')}\n"
        f"Process ended:{context.get('process_ended', 'N/A')}\n"
        f"Model cut:{context.get('model_cut','N/A')}"
    )
    
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
        ax3.spines['right'].set_position(('outward', 60))  
        color = 'tab:green'
        ax3.set_ylabel('Number of Symbols', color=color)
        ax3.plot(layer_names, num_symbols, marker='s', color=color, label='Number of Symbols')
        ax3.tick_params(axis='y', labelcolor=color)
    
    plt.title('Memory Gain Percentage and Cumulative Computation Time by Layer')
    fig.tight_layout()
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(0.03, 0.85), title="Context", labels=[context_info])
    
    return fig, context

def load_json(filename):
    """
    Load JSON data from a file.

    Args:
        filename (str): The path to the JSON file.

    Returns:
        dict: The loaded JSON data.
    """
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def generate_graphs_from_folder(folder_path, output_pdf):
    """
    Generate graphs from JSON files in a folder and save them to a PDF.

    Args:
        folder_path (str): The path to the folder containing JSON files.
        output_pdf (str): The path to the output PDF file.

    Returns:
        None
    """
    pdf_pages = PdfPages(output_pdf)
    
    context_execution_times = {}
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            json_data = load_json(file_path)
            fig, context = process_model_data(json_data)
            
            context_key = (
                context.get('model_name', 'N/A'),
                context.get('input_tensor_size', 'N/A'),
                context.get('num_workers', 0),
                context.get('available_RAM', 0),
                context.get('device', 'N/A'),
                context.get('add_symbol', 'N/A'),
                context.get('renew_abstract_domain', 'N/A'),
                context.get('verbose', 'N/A'),
                context.get('num_symbols', 'N/A'),
                context.get('model_cut','N/A')
            )
            
            context_key = tuple(tuple(item) if isinstance(item, list) else item for item in context_key)
            
            noise_level = context.get('noise_level', 'N/A')
            cumulative_time = sum([layer['computation_time'] for layer in json_data['layers']])
            
            if context_key not in context_execution_times:
                context_execution_times[context_key] = []
            
            context_execution_times[context_key].append((noise_level, cumulative_time))
            
            pdf_pages.savefig(fig)
            plt.close(fig)
    
    for context_key, execution_times in context_execution_times.items():
        execution_times.sort(key=lambda x: x[0])  
        noise_levels, cumulative_times = zip(*execution_times)
        
        plt.figure(figsize=(10, 6))
        plt.plot(noise_levels, cumulative_times, marker='o', color='tab:red')
        plt.xlabel('Noise Level')
        plt.ylabel('Cumulative Computation Time (s)')
        plt.title('Cumulative Computation Time vs Noise Level')
        
        context_info = (
            f"Model Name: {context_key[0]}\n"
            f"Input tensor shape: {context_key[1]}\n"
            f"Workers: {context_key[2]}\n"
            f"RAM: {context_key[3]} GB\n"
            f"Device: {context_key[4]}\n"
            f"Add Symbol: {context_key[5]}\n"
            f"Renew Abstract Domain: {context_key[6]}\n"
            f"Verbose: {context_key[7]}\n"
            f"Num Symbols: {context_key[8]}\n"
            f"Model cut:{context_key[9]}"
        )
        
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Context", labels=[context_info])       
        plt.tight_layout()
        plt.grid(True)
        
        pdf_pages.savefig()
        plt.close()
    
    pdf_pages.close()

def load_json(filename):
    """
    Load JSON data from a file.

    Args:
        filename (str): The path to the JSON file.

    Returns:
        dict: The loaded JSON data.
    """
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def generate_graphs_from_folder(folder_path, output_pdf):
    """
    Generate graphs from JSON files in a folder and save them to a PDF.

    Args:
        folder_path (str): The path to the folder containing JSON files.
        output_pdf (str): The path to the output PDF file.

    Returns:
        None
    """
    pdf_pages = PdfPages(output_pdf)
    
    context_execution_times = {}
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            json_data = load_json(file_path)
            fig, context = process_model_data(json_data)
            
            context_key = (
                context.get('model_name', 'N/A'),
                context.get('input_tensor_size', 'N/A'),
                context.get('num_workers', 0),
                context.get('available_RAM', 0),
                context.get('device', 'N/A'),
                context.get('add_symbol', 'N/A'),
                context.get('renew_abstract_domain', 'N/A'),
                context.get('verbose', 'N/A'),
                context.get('num_symbols', 'N/A')
            )
            
            context_key = tuple(tuple(item) if isinstance(item, list) else item for item in context_key)
            
            noise_level = context.get('noise_level', 'N/A')
            cumulative_time = sum([layer['computation_time'] for layer in json_data['layers']])
            
            if context_key not in context_execution_times:
                context_execution_times[context_key] = []
            
            context_execution_times[context_key].append((noise_level, cumulative_time))
            
            pdf_pages.savefig(fig)
            plt.close(fig)
    
    for context_key, execution_times in context_execution_times.items():
        execution_times.sort(key=lambda x: x[0])  
        noise_levels, cumulative_times = zip(*execution_times)
        
        plt.figure(figsize=(10, 6))
        plt.plot(noise_levels, cumulative_times, marker='o', color='tab:red')
        plt.xlabel('Noise Level')
        plt.ylabel('Cumulative Computation Time (s)')
        plt.title('Cumulative Computation Time vs Noise Level')
        
        context_info = (
            f"Model Name: {context_key[0]}\n"
            f"Input tensor shape: {context_key[1]}\n"
            f"Workers: {context_key[2]}\n"
            f"RAM: {context_key[3]} GB\n"
            f"Device: {context_key[4]}\n"
            f"Add Symbol: {context_key[5]}\n"
            f"Renew Abstract Domain: {context_key[6]}\n"
            f"Verbose: {context_key[7]}\n"
            f"Num Symbols: {context_key[8]}"
        )
        
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Context", labels=[context_info])
        plt.tight_layout()
        plt.grid(True)
        
        pdf_pages.savefig()
        plt.close()
    
    pdf_pages.close()

# Folder path containing JSON files
folder_path = os.path.abspath(__file__)
folder_path = os.path.dirname(folder_path)
# Output PDF file
output_pdf = os.path.join(folder_path, 'output.pdf')

# Generate the graphs and save them to a PDF
generate_graphs_from_folder(folder_path, output_pdf)
