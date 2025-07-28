# Code for CORTEX: COunterfactual Reasoning for Temporal EXplanations

This repository contains the code implementation of CORTEX, a novel methodology for generating temporal graph counterfactuals proposed in the paper "COunterfactual Reasoning for Temporal EXplanations: Plausible and Stable Explanations for Time Series Data".

CORTEX leverages a search-based approach designed to ensure **validity** and **plausibility** of the generated explanations, while also guaranteeing **temporal stability**.

We also include a case study on seizure classification from EEG signals to evaluate the framework. 🧠


## How to run an experiment

To run an experiment, do the following steps 🚀:
1. Browse the JSON files within the subfolders of `EEG_data` (or generate them with notebooks `lab\00 - loop generation EEG data CHB MIT` or `lab\00 - loop generation EEG data SIENA`), select the desired records, and move their corresponding files into the main `EEG_data\` folder.
2. Run the notebook `01 - run experiment.ipynb`, specifying the set of recordings (`observations`) you want to use.
3. Execute the notebook `02 - get evaluation metrics.ipynb` to assess the performance of both the oracle and the explainer.
4. Use the notebook `03 - plot top k counterfactuals` to visualize counterfactual explanations on the electrode map for a single record.
  

> **⚠️ Notice:**   
> The notebooks `00 - ...` must be run in the **EEG** environment, while the other notebooks must be run in the **GRETEL** environment.
