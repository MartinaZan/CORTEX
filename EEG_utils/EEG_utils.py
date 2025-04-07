import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

import pickle
import json
import mne  # To read edf files
import re   # For regular expression matching in patient-summary.txt

################################################################################

class FilePatient:
    def __init__(self, root_folder, patient_id, record_id):
        self.root_folder = root_folder
        self.patient_id = patient_id
        self.record_id = record_id

    def get_record_path(self):
        # Returns the full path to the record's file
        return f"{self.root_folder}{self.patient_id}\\{self.patient_id}_{self.record_id}.edf"

    def get_summary_path(self):
        # Returns the full path to the summary file
        return f"{self.root_folder}{self.patient_id}\\{self.patient_id}-summary.txt"
    
################################################################################

class Patient:
    def __init__(self, file_patient: FilePatient, num_points=1500, num_node_features=1, lag_nodes=1, quantile_edges=0.25, corr_sec=10):
        self.file_patient = file_patient
        self.dictionary_unique_pairs = {}

        self.patient_info = {
            "start_time": None,     # Start recording
            "end_time": None,       # End recording
            "seizure_starts": [],   # Seizure start times
            "seizure_ends": []      # Seizure end times
        }

        self.num_node_features = num_node_features

        self.num_seizures = None
        self.df = None

        self.frequency = 256

        self.num_points = num_points                    # Number of points for each class
        self.lag_corr = int(self.frequency*corr_sec)    # Number of lag for correlation calculation (corr_sec seconds)
        self.lag_nodes = lag_nodes                      # Distance of lags for node features
        self.buffer_time = None                         # Buffer to skip data too close to beginning and end of seizures
        self.skip_0 = None
        self.skip_1 = None

        self.indices = None

        self.quantile_edges = quantile_edges    # Quantile for edge selection


    def set_skip_values(self):
        length_recording = self.get_length_recording()
        length_seizures = self.get_length_seizures()

        self.buffer_time = int(min(length_seizures) / 10 * self.frequency) # In this way I discard 20% of the seizures points

        self.skip_0 = int(((length_recording - self.lag_corr / self.frequency - self.buffer_time / self.frequency * len(length_seizures) * 2) / self.num_points) * self.frequency)
        self.skip_1 = int(((sum(length_seizures) - self.buffer_time / self.frequency * len(length_seizures) * 2) / self.num_points) * self.frequency)
        

    def get_times(self):
        """Function to extract times of the recording"""

        # Start = int(self.get_times()[0])
        # End = int(self.get_times()[-1])

        return np.array(self.df.index)
    
    
    def get_data(self):
        """Function to extract data of the recording"""
        return self.df.values.T


    def get_length_recording(self):
        return int(self.get_times()[-1]) - int(self.get_times()[0])
    

    def get_length_seizures(self):
        return [self.patient_info["seizure_ends"][i] - self.patient_info["seizure_starts"][i] for i in range(len(self.patient_info["seizure_starts"]))]


    def extract_seizure_info(self):
        """Function to extract seizure information from the summary file"""

        # Record to find in patient-summary.txt
        record_to_find = f"{self.file_patient.patient_id}_{self.file_patient.record_id}.edf"

        # Open the patient-summary.txt file
        with open(self.file_patient.get_summary_path(), "r") as file:
            data = file.read()

        # Split the file content into blocks to create a list. Each element of the list starts
        # with "File Name: "
        file_blocks = re.split(r"(?=File Name:)", data)

        # Iteration over blocks to find the correct one
        for block in file_blocks:
            if record_to_find in block:
                # Data extraction
                self.patient_info["start_time"] = re.search(r"File Start Time:\s*([\d:]+)", block).group(1)
                self.patient_info["end_time"] = re.search(r"File End Time:\s*([\d:]+)", block).group(1)
                self.num_seizures = int(re.search(r"Number of Seizures in File:\s*(\d+)", block).group(1))

                # If there are seizures, extract their details
                for i in range(1, self.num_seizures + 1):
                    if self.num_seizures == 1:
                        seizure_start = int(re.search(rf"Seizure Start Time:\s*(\d+)\s*seconds", block).group(1))
                        seizure_end = int(re.search(rf"Seizure End Time:\s*(\d+)\s*seconds", block).group(1))
                        self.patient_info["seizure_starts"].append(seizure_start)
                        self.patient_info["seizure_ends"].append(seizure_end)
                    else:
                        seizure_start = int(re.search(rf"Seizure {i} Start Time:\s*(\d+)\s*seconds", block).group(1))
                        seizure_end = int(re.search(rf"Seizure {i} End Time:\s*(\d+)\s*seconds", block).group(1))
                        self.patient_info["seizure_starts"].append(seizure_start)
                        self.patient_info["seizure_ends"].append(seizure_end)

                return
        else:
            raise FileNotFoundError(f"Record {record_to_find} not found in summary.")

    def create_unique_pair_dict(self, raw_data):
        """Function to create a dictionary of unique channel pairs"""
        unique_pairs = set()    # Using a set to avoid duplicate pairs
        unique_dict = {}        # Dictionary to store unique pairs

        for idx, item in enumerate(raw_data.ch_names):
            if '-' in item:
                elements = item.split('-')

                # Skip if the element does not contain two parts
                if len(elements) < 2 or not elements[0] or not elements[1]:
                    continue

                # Consider both normal and reversed order
                pair_normal = f"{elements[0]}-{elements[1]}"
                pair_inverted = f"{elements[1]}-{elements[0]}"

                # Adds only a unique version to the dictionary
                if pair_normal not in unique_pairs and pair_inverted not in unique_pairs:
                    unique_pairs.add(pair_normal)
                    unique_dict[idx] = pair_normal

        self.dictionary_unique_pairs = unique_dict

    
    def load_data(self):
        """Function to load and preprocess EEG data"""

        # Load EEG data
        raw_data = mne.io.read_raw_edf(self.file_patient.get_record_path())

        # Create dictionary of unique channel pairs
        self.create_unique_pair_dict(raw_data)

        # Extraction of EEG data and times
        data, times = raw_data[:]

        # Remove duplicate signals
        data = data[list(self.dictionary_unique_pairs.keys())]

        # Scaling of the signals
        data = 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1

        # Set start and end point of the record
        Start = max([0, min(self.patient_info["seizure_starts"]) - 500])
        End = min([(datetime.strptime(self.patient_info["end_time"], "%H:%M:%S") - datetime.strptime(self.patient_info["start_time"], "%H:%M:%S")).seconds, max(self.patient_info["seizure_starts"]) + 500])
        
        data = data[:, (Start * self.frequency):(End * self.frequency)]
        times = times[(Start * self.frequency):(End * self.frequency)]

        # Create DataFrame
        self.df = pd.DataFrame(data.T, columns=[f'c_{i}' for i in range(data.shape[0])])
        self.df.index = times

        self.set_skip_values()


    def plot_signals(self):
        """Plot EEG signals with seizure regions highlighted"""
        data = self.get_data()

        offset = np.max(np.abs(data)) * 2  # Vertical offset for signals

        plt.figure(figsize=(7,5))
        for i in range(len(data)):
            plt.plot(self.get_times(), data[i, :] + i * offset)
            plt.axhline(y=i * offset, color='gray', linestyle='--', linewidth=0.25)

        for start, end in zip(self.patient_info["seizure_starts"], self.patient_info["seizure_ends"]):
            plt.axvspan(start, end, color='yellow', alpha=0.5)

        plt.title(f"Number of seizures: {self.num_seizures}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Channels")
        plt.yticks([i * offset for i in range(len(data))], [f'c_{i}' for i in range(len(data))])
        plt.xlim((self.df.index[0], self.df.index[-1]))
        plt.tight_layout()
        plt.show()


    def compute_indices(self):
        Start = int(self.get_times()[0])
        End = int(self.get_times()[-1])
        
        indices = list(range(self.lag_corr, (self.patient_info["seizure_starts"][0] - Start) * self.frequency - self.buffer_time, self.skip_0))  # Before first seizure

        for i in range(len(self.patient_info["seizure_starts"])):
            indices += list(range((self.patient_info["seizure_starts"][i] - Start) * self.frequency + self.buffer_time, (self.patient_info["seizure_ends"][i] - Start) * self.frequency - self.buffer_time, self.skip_1))
            
            if i < len(self.patient_info["seizure_starts"]) - 1:
                indices += list(range((self.patient_info["seizure_ends"][i] - Start) * self.frequency + self.buffer_time, (self.patient_info["seizure_starts"][i + 1] - Start) * self.frequency - self.buffer_time, self.skip_0))

        # After last seizure
        indices += list(range((self.patient_info["seizure_ends"][-1] - Start) * self.frequency + self.buffer_time, (End - Start) * self.frequency, self.skip_0))

        self.indices = indices


    def plot_indices(self, xlim = None):
        Start = int(self.get_times()[0])

        xx = [k/self.frequency + Start for k in self.indices]

        plt.figure(figsize=(30,2))
        plt.scatter(xx, np.zeros(len(xx)), s=5)

        for start, end in zip(self.patient_info["seizure_starts"], self.patient_info["seizure_ends"]):
            plt.axvspan(start, end, color='yellow', alpha=0.5)

        if xlim != None:
            plt.xlim(xlim)
        plt.tight_layout()
        plt.show()

################################################################################

def create_graph(patient):
    df = patient.df
    indices = patient.indices
    lag_corr = patient.lag_corr
    lag_nodes = patient.lag_nodes
    seizure_starts = patient.patient_info["seizure_starts"]
    seizure_ends = patient.patient_info["seizure_ends"]
    quantile_edges = patient.quantile_edges

    # Creation of series from the dataframe
    columns = [df.iloc[:, i].to_numpy() for i in range(0,len(df.columns))]
    series = list(map(list, zip(*columns)))
    
    # Definition of node IDs and pairs for edges
    node_ids = {f"c_{channel}": channel for channel in range(0,len(df.columns))}
    pairs = np.array([(i, j) for i in range(len(columns)) for j in range(len(columns))]) # if i != j
    edges = pairs.tolist()

    weights = []        # Edge weights
    edge_list = []      # List of edges
    corr = []           # Correlation values
    node_features = []  # List of node features

    seizure_class = []  # Target (seizure/no seizure)

    num_node_features = patient.num_node_features

    # Extraction of edge weights from correlation matrix
    for k in indices:
        t = df.index[k]
        values_list = []
        new_edges = []

        ##########################################################################

        corr_mat = (df.iloc[(k-lag_corr+1):(k+1)]).corr()

        # Take the absolute values
        corr_mat = np.abs(corr_mat)
        
        # Remove loops
        np.fill_diagonal(corr_mat.values, 0)

        # Keep only highest quantile_edges%
        q = corr_mat.melt().value.quantile(1 - quantile_edges)
        corr_mat[corr_mat < q] = 0

        ##########################################################################

        for i in range(len(edges)):
            row_index = edges[i][0]
            col_index = edges[i][1]

            new_edges.append(edges[i])
            values_list.append(corr_mat.iloc[row_index, col_index])
        
        # Seizure in the sliding window
        seizure = False
        for start, end in zip(seizure_starts, seizure_ends):
            if (start <= t-(lag_corr/patient.frequency)) & (t-(lag_corr/patient.frequency) <= end) | ((start <= t) & (t <= end)):
                seizure = True
        
        if seizure:
            seizure_class.append(1)
        else:
            seizure_class.append(0)
        
        # print(f"k: {k} ---> t: {t} (seizure class: {seizure_class[-1]})")

        corr.append(corr_mat)
        weights.append(values_list)
        edge_list.append(new_edges)
        # node_features.append(series[k])                               # Versione 1 (1 node feature)
        # node_features.append(series[(k-num_node_features+1):(k+1)])   # Versione 2 (n node feature)

        # Versione attuale
        node_features.append(series[(k-(num_node_features-1)*lag_nodes):(k+1):lag_nodes])   # Versione 3 (n node feature laggate)

        ###########################################
        # TENTATIVO (stessa node feature):
        # node_features.append([series[k]] * num_node_features)
        ###########################################

        ###########################################
        # TENTATIVO (node feature che decadono nel tempo):
        # prova = series[(k-(num_node_features-1)*lag_nodes):(k+1):lag_nodes]
        # pesi = (np.linspace(0, 1, num_node_features)) # np.linspace(1/(num_node_features/4),1, num_node_features)
        # for i in range(len(prova)):
        #     prova[i] = list(np.array(prova[i]) * pesi[i])
        # node_features.append(prova)
        ###########################################

    return node_ids, corr, weights, edge_list, node_features, seizure_class

################################################################################

def export_data_to_GRETEL(patient):
    node_ids, _, weights, edge_list, node_features, seizure_class = create_graph(patient)

    #
    ## Saving graph into json file
    #
    tempi = list(range(len(seizure_class)))
    index_dict = dict(zip(tempi, edge_list))
    weight_dict = dict(zip(tempi, weights))
    edge_dict = {
        "edge_index": index_dict,
        "edge_weight": weight_dict
    }

    dictionary = {
        "patient": patient.file_patient.patient_id,
        "record": patient.file_patient.record_id,
        "edge_mapping": edge_dict,
        "node_ids": node_ids,
        "time_periods": tempi,
        "target": seizure_class,
        "series": node_features
    }

    json_object = json.dumps(dictionary)
    file_path = f"EEG_data\dataset_{patient.file_patient.patient_id}_{patient.file_patient.record_id}.json"

    with open(file_path, "w") as outfile:
        outfile.write(json_object)

    #
    ## Saving useful variables into pkl file
    #
    save_variables = {
        "indici": patient.indices,
        "Start": int(patient.get_times()[0]),
        "End": int(patient.get_times()[-1]),
        "seizure_starts": patient.patient_info["seizure_starts"],
        "seizure_ends": patient.patient_info["seizure_ends"],
        "seizure_class": seizure_class
    }

    with open("..\\..\\explainability\\GRETEL-repo\\EEG_data\\" + f"EEG_data_params_{patient.file_patient.patient_id}_{patient.file_patient.record_id}.pkl", 'wb') as f:
        pickle.dump(save_variables, f)

################################################################################

def export_coordinates(patient):
    electrode_coordinates = {
        'A1': (25, 460), 'A2': (645, 455), 'AF3': (260, 185), 'AF4': (420, 185), 'AF7': (205, 175),  'AF8': (475, 175), 'AFZ': (340, 185), 'C1': (285, 380), 'C2': (395, 380),
        'C3': (230, 380), 'C4': (450, 380), 'C5': (175, 380), 'C6': (505, 380), 'CP1': (290, 445), 'CP2': (390, 445), 'CP3': (235, 445), 'CP4': (445, 445), 'CP6': (495, 450),
        'CP7': (185, 450), 'CPZ': (340, 445), 'CZ': (340, 380), 'F1': (295, 250), 'F10': (575, 210), 'F2': (385, 250), 'F3': (250, 245), 'F4': (430, 245), 'F5': (205, 240),
        'F6': (475, 240), 'F7': (160, 235), 'F8': (520, 235), 'F9': (100, 210), 'FC1': (290, 315), 'FC2': (390, 315), 'FC3': (235, 315), 'FC4': (445, 315), 'FC5': (185, 310),
        'FC6': (495, 310), 'FCZ': (340, 315), 'FP1': (270, 135), 'FP2': (410, 135), 'FPZ': (340, 120), 'FT7': (130, 305), 'FT8': (550, 305), 'FT9': (85, 285), 'FT10': (587, 285),
        'FZ': (340, 253), 'IZ': (340, 685), 'NZ': (340, 65), 'O1': (270, 625), 'O2': (410, 625), 'OZ': (340, 640), 'P1': (295, 510), 'P10': (560, 555), 'P2': (385, 510),
        'P3': (250, 515), 'P4': (430, 515), 'P5': (205, 520), 'P6': (475, 520), 'P7': (160, 525), 'P8': (520, 525), 'P9': (110, 555), 'PO3': (260, 575), 'PO4': (420, 575),
        'PO7': (205, 585), 'PO8': (475, 585), 'POZ': (340, 575), 'PZ': (340, 507), 'T10': (607, 380), 'T7': (120, 380), 'T8': (560, 380), 'T9': (73, 380), 'TP10': (600, 475),
        'TP7': (130, 455), 'TP8': (550, 455), 'TP9': (75, 475)
    }

    # List of mid points coordinates
    punti_medi = []

    img = plt.imread("EEG_utils\\nodi-vuoto.png")

    fig, ax = plt.subplots(figsize=(10,6))
    ax.imshow(img)

    for c, coppia in patient.dictionary_unique_pairs.items():
        # Pair of electrodes
        punto = coppia.split('-')

        if len(punto) < 2:
            continue

        # Plot style
        mid_line_width = fig.get_size_inches()[0] / 10
        mid_marker_size = fig.get_size_inches()[0] / 2
        marker_size = fig.get_size_inches()[0] * 2
        font_size = fig.get_size_inches()[0] / 1.5
        marker_edge_width = fig.get_size_inches()[0] / 8
        
        # Coordinates of the two elettrodes
        p1 = electrode_coordinates[punto[0]]
        p2 = electrode_coordinates[punto[1]]

        # Coordinates of the mid point
        mid_point = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        punti_medi.append(mid_point)

        # Draw line between the two electrodes and square at the midpoint
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=mid_line_width)
        ax.plot(mid_point[0], mid_point[1], 's', markersize=mid_marker_size, color='blue')
        ax.text(mid_point[0], mid_point[1], "c_" + str(c), fontsize=font_size, ha='center', va='center')

        # Draw the nodes corresponding to the electrodes
        ax.text(p1[0], p1[1], punto[0], fontsize=font_size, ha='center', va='center') #, bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))
        ax.plot(p1[0], p1[1], 'wo', markersize=marker_size, markeredgecolor='black', markeredgewidth=marker_edge_width)
        ax.text(p2[0], p2[1], punto[1], fontsize=font_size, ha='center', va='center') #, bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))
        ax.plot(p2[0], p2[1], 'wo', markersize=marker_size, markeredgecolor='black', markeredgewidth=marker_edge_width)

    plt.axis('off')
    plt.show()

    # Save a dictionary with coordinates of midpoints (= graph node positions)
    dizionario_punti_medi = {i: punti_medi[i] for i in range(len(punti_medi))}

    with open("..\\..\\explainability\\GRETEL-repo\\EEG_data\\" + f"mid_points_{patient.file_patient.patient_id}_{patient.file_patient.record_id}.pkl", 'wb') as f:
        pickle.dump(dizionario_punti_medi, f)
