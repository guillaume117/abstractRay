import os
import json

def save_results_to_json(timestart, json_file_prefix,all_results):
        path_script = os.path.abspath(__file__)
        folder_script = os.path.dirname(path_script)
        folder_result = os.path.join(folder_script,json_file_prefix)
        print(folder_result)
        if not os.path.exists(folder_result):
            os.makedirs(folder_result)

        json_file = f'{json_file_prefix}_time_eval_{timestart}.json'
        json_file = os.path.join(folder_result,json_file)
        with open(json_file, 'w') as f:
            json.dump(all_results, f, indent=4, default=str) 