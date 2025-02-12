import os
import json

def save_results_to_json(timestart, json_file_prefix, all_results):
    """
    Save the evaluation results to a JSON file.

    This function saves the provided results to a JSON file. The file is saved in a directory named after
    `json_file_prefix`, and the filename includes the `timestart` to ensure uniqueness.

    Args:
        timestart (float): The timestamp when the evaluation started.
        json_file_prefix (str): The prefix for the JSON file name.
        all_results (dict): The results to be saved in the JSON file.

    Returns:
        None
    """
    path_script = os.path.abspath(__file__)
    folder_script = os.path.dirname(path_script)
    folder_result = os.path.join(folder_script, json_file_prefix)
 
    if not os.path.exists(folder_result):
        os.makedirs(folder_result)

    json_file = f'{json_file_prefix}_time_eval_{timestart}.json'
    json_file = os.path.join(folder_result, json_file)
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=4, default=str)
