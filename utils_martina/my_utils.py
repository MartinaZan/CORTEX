import os
import re
import ast
import json
import shutil

################################################################################################################

def get_most_recent_file(folder_path):
    # Get the name of the most recent file in folder_path

    return max(os.listdir(folder_path), key=lambda f: os.path.getmtime(os.path.join(folder_path, f)))

################################################################################################################

def create_dataset_json(observations):
    # Set dataset json files based on the name of the record

    data_file_names = [f"grafo_corr_sliding_{i}.json" for i in range(len(observations))]

    for i in range(len(observations)):
        shutil.copyfile(f"EEG_data/dataset_{observations[i]}.json", f"data/datasets/gcs/{data_file_names[i]}")

    file_json = {
        "class": "src.dataset.dataset_base.Dataset",
        "parameters": {
            "generator": {
                "class": "src.dataset.generators.gcs.GCS",
                "parameters": {
                    "data_dir": "data/datasets/gcs/",
                    "data_file_name": data_file_names,
                    "data_label_name": "y"
                }
            }
        }
    }

    with open("config\snippets\datasets\GCS.json", "w") as f:
        json.dump(file_json, f, indent=4)

################################################################################################################

def get_embeddings_and_outputs_all(eval_manager):
    # Get embeddings of all elements

    result = []
    expl = eval_manager._evaluators[0]._explainer
    
    for element in expl.dataset.instances:
        result.append(expl.oracle.get_gcn_embeddings(element))
    
    return result

def get_embeddings_and_outputs_test(eval_manager):
    # Get embeddings for the test set

    result = []
    expl = eval_manager._evaluators[0]
    ids_list = [explanation.id for explanation in eval_manager._evaluators[0].explanations]

    for element in expl.dataset.instances:
        if element.id in ids_list:
            result.append(expl._explainer.oracle.get_gcn_embeddings(element))

    return result

################################################################################################################

def extract_dictionary_from_runtime_metric(content,metric):
    # Search for info on the chosen metric
    pattern = metric + "': (\[\{.*?\}\])"
    match = re.search(pattern, content)

    if match:
        # Extract metric information
        data = match.group(1)

        # Convert the string to a list of dictionaries
        list = ast.literal_eval(data)

        # Create a dictionary with 'id' as the key and 'value' as the value
        dict = {item['id']: item['value'] for item in list}

        return dict
    
def extract_time_and_record(content):
    # Search for info on the chosen metric
    pattern = "RuntimeMetric" + "': (\[\{.*?\}\])"
    match = re.search(pattern, content)

    if match:
        # Extract metric information
        data = match.group(1)

        # Convert the string to a list of dictionaries
        list = ast.literal_eval(data)

        # Create a dictionary
        dict_time = {item['id']: item['time'] for item in list}
        dict_record = {item['id']: item['record'] for item in list}

        return dict_time, dict_record