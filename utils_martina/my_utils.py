import os
import re
import ast

################################################################################################################

def get_most_recent_file(folder_path):
    """
    Ricava il nome del file più recente in folder_path
    """
    return max(os.listdir(folder_path), key=lambda f: os.path.getmtime(os.path.join(folder_path, f)))

################################################################################################################

def get_embeddings_and_outputs_all(eval_manager):
    result = []

    expl = eval_manager._evaluators[0]._explainer
    
    for element in expl.dataset.instances:
        result.append(expl.oracle.get_gcn_embeddings(element))
    
    return result

def get_embeddings_and_outputs_test(eval_manager):
    result = []

    expl = eval_manager._evaluators[0]
    ids_list = [explanation.id for explanation in eval_manager._evaluators[0].explanations]

    for element in expl.dataset.instances:
        if element.id in ids_list:
            result.append(expl._explainer.oracle.get_gcn_embeddings(element))

    return result

################################################################################################################

def estrai_dizionario_da_runtime_metric(content,metrica):
    # Cerca le info sulla metrica scelta
    pattern = metrica + "': (\[\{.*?\}\])"
    match = re.search(pattern, content)

    if match:
        # Estrae le informazioni sulla metrica
        data = match.group(1)

        # Converte la stringa in una lista di dizionari
        list = ast.literal_eval(data)

        # Crea un dizionario con 'id' come chiave e 'value' come valore
        dict = {item['id']: item['value'] for item in list}

        return dict
    
def estrai_time_e_record(content):
    # Cerca le info sulla metrica scelta
    pattern = "RuntimeMetric" + "': (\[\{.*?\}\])"
    match = re.search(pattern, content)

    if match:
        # Estrae le informazioni sulla metrica
        data = match.group(1)

        # Converte la stringa in una lista di dizionari
        list = ast.literal_eval(data)

        # Crea un dizionario con 'id' come chiave e 'value' come valore
        dict_time = {item['id']: item['time'] for item in list}
        dict_record = {item['id']: item['record'] for item in list}

        return dict_time, dict_record