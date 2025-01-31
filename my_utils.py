import os

################################################################################################################à

def get_most_recent_file(folder_path):
    """
    Ricava il nome del file più recente in folder_path
    """
    return max(os.listdir(folder_path), key=lambda f: os.path.getmtime(os.path.join(folder_path, f)))

################################################################################################################à

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

################################################################################################################à

