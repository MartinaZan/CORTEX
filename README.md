# GRETEL-repo

This is a fork of the [GRETEL](https://github.com/aiim-research/GRETEL) repository, specifically developed for **temporal graph counterfactual explanation** and including a case study based on seizure classification from EEG signals. 🧠

## How to run an experiment

To run an experiment, use these notebooks (in order) 🚀:
- `0 - EEG data CHB MIT.ipynb` or `0 - EEG data SIENA.ipynb` (depending on the dataset): Choose the record to study and generate the JSON dataset.
- `1 - run experiment.ipynb`: Run the GRETEL experiment using a set of recordings (`observations`) based on the configuration files.
- `2 - plot counterfactuals.ipynb`: Visualize counterfactual explanations on the electrode map for a single record.
- `3 - get oracle info.ipynb`: Display information related to the oracle (accuracy and embeddings, if applicable).
- `4 - plot results.ipynb`: Plot oracle and explainer metrics over time for a single record.  
  

> **⚠️ Notice:**   
> The notebooks `0 - ...` must be run in the **EEG** environment, while the other notebooks must be run in the **GRETEL** environment.
