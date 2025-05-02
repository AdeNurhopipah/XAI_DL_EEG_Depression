"""
This module provides functions to handle synthetic EEG data for classification tasks, including loading, preprocessing, and visualisation.

Functions:
    - GetSyntheticData: Loads synthetic EEG data and ground truth for specified domain and class.
    - LoadSyntheticData: Loads multiple classes of synthetic EEG data and concatenates them.
    - MaxAbsNorm: Normalizes data using the maximum absolute value.
    - NormData: Applies MaxAbsNorm to all data samples.
    - ShuffledData: Shuffles the data and labels randomly.
    - GetRandLables: Generates random labels for class data.
    - LblEncoding: Encodes labels using one-hot encoding.
    - GetTaperredGt: Tappered ground truth data.
    - WindowingEEg: Segments EEG data into non-overlapping windows for analysis or model input
    - NotchFiltering: Removes specific frequency components (power line noise at 50 Hz) from EEG signals.
    - BandPassFiltering: Filters EEG signals to retain specific frequency interval (0.5Hz to 50Hz).
    - ApplyFastICA: Applies Fast Independent Component Analysis (FastICA) for artifact removal.
    - PlotData: Plots EEG data for a specific class across channels.
    - PlotChannels: Plots individual channel data for an EEG sample.

Author: Ade Nurhopipah
Email: ade.nurhopipah@postgrad.otago.ac.nz
"""

#Dependencies
import numpy as np
from os import path, listdir  
from pandas import read_excel
from matplotlib import pyplot as plt
from mne import io
from scipy import signal
from sklearn.decomposition import FastICA
from sklearn.preprocessing import OneHotEncoder


#Read serega db
def GetSyntheticData(file_path, domain, class_idx):
    class_data, gt_data = [], []

    # Construct file paths
    data_path = path.join(file_path, f'sim_db_{domain}{class_idx}.set')
    gt_path = path.join(file_path, f'gt_db_{domain}{class_idx}.set')

    # Load class data
    if path.exists(data_path):
        print(f'Opening file: {data_path}')
        dt = io.read_epochs_eeglab(data_path)
        class_data.append(dt.get_data())
    else:
        print(f"Data file not found for class {class_idx} at {data_path}")

    # Load ground truth data
    if path.exists(gt_path):
        print(f'Opening ground truth file: {gt_path}')
        gt = io.read_epochs_eeglab(gt_path)
        gt_data.append(gt.get_data())
    else:
        print(f"Ground truth file not found for class {class_idx} at {gt_path}")

    # Concatenate data if files were successfully loaded
    class_data = np.concatenate(class_data, axis=0) if class_data else np.array([])
    gt_data = np.concatenate(gt_data, axis=0) if gt_data else np.array([])

    return class_data, gt_data

#Load data
def LoadSyntheticData(data_path, domain):
    # Initialize lists for data and labels
    data_list, gt_list, label_list = [], [], []
    
    # Load class 1 and class 2 data (these are always included)
    for class_idx in [1, 2]:
        data, gt = GetSyntheticData(file_path=data_path, domain=domain, class_idx=class_idx)
        data_list.append(data)
        gt_list.append(gt)
        label_list.append(np.full(data.shape[0], class_idx - 1))  

    # Load additional classes if in the specified domain
    if domain in ['spectral', 'temporal']:
        for class_idx in [3, 4]:
            data, gt = GetSyntheticData(file_path=data_path, domain=domain, class_idx=class_idx)
            data_list.append(data)
            gt_list.append(gt)
            label_list.append(np.full(data.shape[0], class_idx - 1))  

    # Concatenate all data, ground truth, and labels
    all_data = np.concatenate(data_list, axis=0)
    all_gt = np.concatenate(gt_list, axis=0)
    all_labels = np.concatenate(label_list, axis=0)

    #Set channel names
    channel_names = [
        'Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'CP5',
        'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'CP6',
        'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2',
        'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7',
        'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2',
        'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4',
        'F2', 'Iz'
    ]

    return all_data, all_gt, all_labels, channel_names

# Max absolute normalisation
def MaxAbsNorm(np_data):
    max_val = np.max(np.abs(np_data))
    if max_val == 0:
        print('Warning the data has maximumm value=0. Returning zero numpy')
        return np.zeros_like(np_data)
    else:
      np_data = np_data / max_val

    return np.array(np_data)


# Normalisation 
def NormData (np_data):
    data_norm = []
    for dt in np_data:
        dt_norm =  MaxAbsNorm(dt)
        data_norm.append(dt_norm)

    return np.array(data_norm)


# Shuffled Data
def ShuffledData (np_data, ls_labels, np_gt ):
    np.random.seed(42)
    shuffled_idx = np.random.permutation(len(ls_labels))
    np_data = np_data[shuffled_idx]
    ls_labels = ls_labels[shuffled_idx]
    np_gt = np_gt[shuffled_idx]

    #print('Shuffled Index', shuffled_idx)
    return np_data, ls_labels, np_gt

# Generate random label
def GetRandLabels(ls_lbl):  
    rand_labels = ls_lbl.copy()  
    label_list = np.unique(ls_lbl) 

    # set n random
    num_random = int(len(ls_lbl)) #*0.80) 
    random_indices = np.random.choice(len(ls_lbl), size=num_random, replace=False)
    
    for idx in random_indices:
        rand_labels[idx] = int(np.random.choice(label_list)) 
    return rand_labels

#Encoding labels
def LblEncoding (ls_lbl, num_classes):
    categories = [list(np.arange(num_classes))]
    enc = OneHotEncoder(categories=categories, handle_unknown='ignore')
    label_encoded = enc.fit_transform(ls_lbl.reshape(-1, 1)).toarray()

    return label_encoded

# Get tapered ground truth data
def GetTaperredGt (gt, th):
  tap_gt = np.zeros_like(gt)
  sample = np.abs(gt)
  sample_norm = MaxAbsNorm(sample)
  sample_th = np.where((sample_norm > th), 1, 0)

  for i in range (len(sample_th)):
    window = np.zeros_like(sample_th[i])
    indices = np.where(sample_th[i] == 1)[0]
    if len(indices) > 0:
        first_idx= indices[0]
        last_idx = indices[-1]
        window_tukey = signal.windows.tukey(len(window[first_idx:last_idx+1]),0)
        window[first_idx:last_idx+1] =window_tukey
        tap_gt[i] = window

  return tap_gt

# Windowing real EEG dataset
def WindowingEEg(db_path, label_path, n_chan, sampling_rate, len_window, n_window, rm_seg,
                 abs_norm=False, bandpass_fil=False, notch_fil=False, fastICA_fil=False):
    
    file_list =  listdir(db_path)
    info_label = read_excel(label_path)
    data_dict = {}
    label_dict = {}

    print(f'Windowing eeg data files...')
    for file in file_list:
        file_path = f'{db_path}{file}'
        print(f'File path: {file}')
        if path.exists(file_path):
            data_file = np.load(file_path)
            dt_len = data_file.shape[1]
            data_file = data_file[:n_chan, (15*sampling_rate):dt_len-(rm_seg*sampling_rate)]

            if abs_norm:
                print('Data normalisation...')
                data_file = MaxAbsNorm(data_file)
            if bandpass_fil:
                print('Bandpass filtering...')
                data_file = BandPassFiltering(data_file, sampling_rate)
            if notch_fil:
                print('Notch filtering...')
                data_file = NotchFiltering(data_file, sampling_rate)
            if fastICA_fil:
                print('Fast-ICA artefact removal..')
                data_file = ApplyFastICA(data_file, min(n_chan,20))

            status = info_label.loc[info_label['id'] == file, 'status'].values
            if status =='Healthy':
                stat_id = 0
            elif status =='Depression':
                stat_id = 1
            else :
                print(f"id {file} not found!")

            count = 0
            data_window, label_window = [],[]
            for i in range(0, data_file.shape[1], len_window):
                if (i+len_window) > (data_file.shape[1]):
                    print(f"Number of window exceed shape of file {file}. Number of window available= {count-1}")
                    break
                if count != None:
                    if (count == n_window):
                        break
                
                data_window.append(data_file[:, i:i+len_window])
                label_window.append(stat_id)
                count += 1
            
            print(f'Number of window extracted: {count}')
            data_dict[file] = data_window
            label_dict[file] = label_window

        else :
            print(f'File {file} not exist!')
    
    return  data_dict, label_dict

# Apply notch filtering
def NotchFiltering(dt, sampling_rate, f0=50, q=30):
    b, a = signal.iirnotch(f0, q, sampling_rate)
    filtered_data = np.array([signal.filtfilt(b, a, channel) for channel in dt])
    return filtered_data

# Apply bandpass filtering
def BandPassFiltering(dt, sampling_rate, fmin=0.5, fmax=100):
    b, a = signal.butter(4, [fmin / (sampling_rate / 2), fmax / (sampling_rate / 2)], btype='band')
    filtered_data = np.array([signal.filtfilt(b, a, channel) for channel in dt])
    return filtered_data

# Apply fastICA artefact removal
def ApplyFastICA(dt, n_components, reject_indices=[0,1,2]): #decide which component to remove
    ica = FastICA(n_components=n_components, random_state=42)
    components = ica.fit_transform(dt.T).T  
    # Remove specified components
    if reject_indices:
        components[reject_indices, :] = 0  
    # Reconstruct the cleaned data
    clean_data = ica.mixing_.dot(components)
    return clean_data

# Plot eeg sample data
def PlotData(data, class_idx, chan_num):
    fig = plt.figure(figsize=(7, 3))
    for channel_index in range(chan_num):
        plt.plot(data[channel_index], label=f'Channel {channel_index}')
        plt.title(f'Sample of class {class_idx} for all channels')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
    plt.show()

    return fig

#Plot eeg each channel in a sample
def PlotChannels(data, channel_names, chan_num):
    fig, axs = plt.subplots(chan_num, 1, figsize=(7, 1 * chan_num), sharex=True)
    for channel_index in range(chan_num):
        axs[channel_index].plot(data[channel_index])
        axs[channel_index].set_ylabel(channel_names[channel_index])
        axs[channel_index].grid()
    plt.suptitle(f'Sample signal each channels')
    plt.show()

    return fig
