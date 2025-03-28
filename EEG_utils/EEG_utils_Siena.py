import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

import pickle
import json
import mne  # To read edf files
import re   # For regular expression matching in patient-summary.txt

#################################################################################################

def diff_times(start,stop):
    return (datetime.strptime(stop, "%H.%M.%S") - datetime.strptime(start, "%H.%M.%S")).seconds

#################################################################################################

class FilePatient:
    def __init__(self, root_folder, patient_id, record_id):
        self.root_folder = root_folder
        self.patient_id = patient_id
        self.record_id = record_id

    def get_record_path(self):
        # Return the full path to the record's file
        return f"{self.root_folder}{self.patient_id}\\{self.patient_id}-{self.record_id}.edf"

    def get_summary_path(self):
        # Return the full path to the summary file
        return f"{self.root_folder}{self.patient_id}\\Seizures-list-{self.patient_id}.txt"
    
#################################################################################################

class Patient:
    def __init__(self, file_patient: FilePatient, num_points=1500, num_node_features=1):
        self.file_patient = file_patient
        self.dictionary_unique_channels = {}

        self.patient_info = {
            "start_time": None,     # Start recording
            "end_time": None,       # End recording
            "seizure_starts": [],   # Seizure start times
            "seizure_ends": []      # Seizure end times
        }

        self.num_node_features = num_node_features

        self.num_seizures = None
        self.df = None

        self.frequency = 512

        self.num_points = num_points            # Number of points for each class
        self.lag_corr = int(self.frequency*10)  # Number of lag for correlation calculation (10 seconds)
        self.buffer_time = None                 # Buffer to skip data too close to beginning and end of seizures
        self.skip_0 = None
        self.skip_1 = None

        self.indices = None

        self.threshold = 0                      # Threshold for pruning


    def set_skip_values(self):
        length_recording = self.get_length_recording()
        length_seizures = self.get_length_seizures()

        self.buffer_time = int(min(length_seizures) / 10 * self.frequency)  # In this way I discard 20% of the seizures points

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
        record_to_find = f"{self.file_patient.patient_id}-{self.file_patient.record_id}.edf"

        # Open the patient-summary.txt file
        with open(self.file_patient.get_summary_path(), "r") as file:
            data = file.read()

        # Split the file content into blocks to create a list. Each element of the list starts
        # with "File Name: "
        file_blocks = re.split(r"(?=File name:)", data)

        # Iteration over blocks to find the correct one
        for block in file_blocks:
            if record_to_find in block:
                # Data extraction
                start_time = re.search(r"Registration start time:\s*([\d.]+)", block).group(1)
                end_time = re.search(r"Registration end time:\s*([\d.]+)", block).group(1)
                
                seizure_start = re.search(r"Seizure start time:\s*([\d.]+)", block).group(1)
                seizure_end = re.search(r"Seizure end time:\s*([\d.]+)", block).group(1)
                
                # print(start_time)
                # print(end_time)
                # print(seizure_start)
                # print(seizure_end)

                self.patient_info["start_time"] = 0
                self.patient_info["end_time"] = diff_times(start_time,end_time)
                self.patient_info["seizure_starts"].append(diff_times(start_time,seizure_start))
                self.patient_info["seizure_ends"].append(diff_times(start_time,seizure_end))
                
                return
        else:
            raise FileNotFoundError(f"Record {record_to_find} not found in summary.")

    def create_unique_channel_dict(self, raw_data):
        """Function to create a dictionary of unique channels"""
        unique_channels = set()    # Using a set to avoid duplicate pairs
        unique_dict = {}           # Dictionary to store unique pairs

        channels = [item for item in raw_data.ch_names]

        for idx, item in enumerate(raw_data.ch_names):
            if ' ' in item:
                if item.split(' ')[0] != item.split(' ')[1]:
                    element = item.split(' ')[1].upper()

                    # Adds only a unique version to the dictionary
                    if element not in unique_channels:
                        unique_channels.add(element)
                        unique_dict[idx] = element
        
        self.dictionary_unique_channels = unique_dict

    
    def load_data(self):
        """Function to load and preprocess EEG data"""

        # Load EEG data
        raw_data = mne.io.read_raw_edf(self.file_patient.get_record_path())

        # Create dictionary of unique channel pairs
        self.create_unique_channel_dict(raw_data)

        # Extraction of EEG data and times
        data, times = raw_data[:]

        # Remove duplicate signals
        data = data[list(self.dictionary_unique_channels.keys())]

        # Scaling of the signals
        data = 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1

        # Set start and end point of the record
        Start = max([0, min(self.patient_info["seizure_starts"]) - 500])
        End = min([self.patient_info["end_time"] - self.patient_info["start_time"], max(self.patient_info["seizure_starts"]) + 500])
        
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
    seizure_starts = patient.patient_info["seizure_starts"]
    seizure_ends = patient.patient_info["seizure_ends"]

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

        corr_mat = (df.iloc[(k-lag_corr):k]).corr()

        # Take the absolute values
        corr_mat = np.abs(corr_mat)
        
        # Remove loops
        np.fill_diagonal(corr_mat.values, 0)

        # Keep only highest ...%
        q = corr_mat.melt().value.quantile(0.75)
        print(q)
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
        
        print(f"k: {k} ---> t: {t} (seizure class: {seizure_class[-1]})")

        corr.append(corr_mat)
        weights.append(values_list)
        edge_list.append(new_edges)
        # node_features.append(series[k])
        node_features.append(series[(k-num_node_features+1):(k+1)])

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

#################################################################################################

def export_coordinates(patient):
    """
    This function is completely different from that of the CHB-MIT dataset, because in the
    CHB-MIT dataset the channels are associated with pairs of electrodes, whereas here they
    are associated with individual electrodes.
    """

    # From wikipedia (https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG)):
    # - T3 is now T7
    # - T4 is now T8
    # - T5 is now P7
    # - T6 is now P8
    # (maybe also CP5 and CP7 coincide...)

    electrode_coordinates = {
        'A1': (25, 460), 'A2': (645, 455), 'AF3': (260, 185), 'AF4': (420, 185), 'AF7': (205, 175),  'AF8': (475, 175), 'AFZ': (340, 185), 'C1': (285, 380), 'C2': (395, 380),
        'C3': (230, 380), 'C4': (450, 380), 'C5': (175, 380), 'C6': (505, 380), 'CP1': (290, 445), 'CP2': (390, 445), 'CP3': (235, 445), 'CP4': (445, 445), 'CP6': (495, 450),
        'CP7': (185, 450), 'CPZ': (340, 445), 'CZ': (340, 380), 'F1': (295, 250), 'F10': (575, 210), 'F2': (385, 250), 'F3': (250, 245), 'F4': (430, 245), 'F5': (205, 240),
        'F6': (475, 240), 'F7': (160, 235), 'F8': (520, 235), 'F9': (100, 210), 'FC1': (290, 315), 'FC2': (390, 315), 'FC3': (235, 315), 'FC4': (445, 315), 'FC5': (185, 310),
        'FC6': (495, 310), 'FCZ': (340, 315), 'FP1': (270, 135), 'FP2': (410, 135), 'FPZ': (340, 120), 'FT7': (130, 305), 'FT8': (550, 305), 'FT9': (85, 285), 'FT10': (587, 285),
        'FZ': (340, 253), 'IZ': (340, 685), 'NZ': (340, 65), 'O1': (270, 625), 'O2': (410, 625), 'OZ': (340, 640), 'P1': (295, 510), 'P10': (560, 555), 'P2': (385, 510),
        'P3': (250, 515), 'P4': (430, 515), 'P5': (205, 520), 'P6': (475, 520), 'P7': (160, 525), 'T5': (160, 525), 'P8': (520, 525), 'T6': (520, 525), 'P9': (110, 555), 'PO3': (260, 575), 'PO4': (420, 575),
        'PO7': (205, 585), 'PO8': (475, 585), 'POZ': (340, 575), 'PZ': (340, 507), 'T10': (607, 380), 'T7': (120, 380), 'T3': (120, 380), 'T8': (560, 380), 'T4': (560, 380), 'T9': (73, 380), 'TP10': (600, 475),
        'TP7': (130, 455), 'TP8': (550, 455), 'TP9': (75, 475), 'CP5': (185, 450)
    }

    # List of coordinates
    punti = []

    img = plt.imread("EEG_utils\\nodi-vuoto.png")

    fig, ax = plt.subplots(figsize=(10,6))
    ax.imshow(img)

    for c, punto in patient.dictionary_unique_channels.items():

        mid_line_width = fig.get_size_inches()[0] / 10
        mid_marker_size = fig.get_size_inches()[0] / 2
        marker_size = fig.get_size_inches()[0] * 2
        font_size = fig.get_size_inches()[0] / 1.5
        marker_edge_width = fig.get_size_inches()[0] / 8
        
        p1 = electrode_coordinates[punto]
        punti.append(p1)

        # Draw the nodes
        ax.text(p1[0], p1[1], punto, fontsize=font_size, ha='center', va='center') #, bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))
        ax.plot(p1[0], p1[1], 'wo', markersize=marker_size, markeredgecolor='black', markeredgewidth=marker_edge_width)

    plt.axis('off')
    plt.show()

    # Save a dictionary with graph node coordinates
    dizionario_punti = {i: punti[i] for i in range(len(punti))}

    with open("..\\..\\explainability\\GRETEL-repo\\EEG_data\\" + f"mid_points_{patient.file_patient.patient_id}_{patient.file_patient.record_id}.pkl", 'wb') as f:
        pickle.dump(dizionario_punti, f)
