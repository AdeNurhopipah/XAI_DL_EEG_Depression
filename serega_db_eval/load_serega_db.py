"""
This script loads the Serega EEG synthetic dataset, normalises the data, shuffles it with labels and ground truth,
and then saves the processed data and channel names as .npy files for future use. It also includes functionality to display sample
EEG data before and after normalisation.

"""

# Dependencies 
from argparse import ArgumentParser
from numpy import array, save, min, max, random

from sys import path
# provide suitable link
path.append('/home/ade/eeg_xai_code/utils')
from eeg_dbf import LoadSyntheticData, NormData, ShuffledData, GetRandLabels, PlotData, PlotChannels

# Parameter setup
parser = ArgumentParser()
parser.add_argument('--domain', type=str, required=False, choices=['temporal', 'spectral', 'spatial'], default="temporal")
args = parser.parse_args()
domain = args.domain 
snr =  "-3.5"

# Provide data_path
data_path = f'eeg_xai_code/serega_eval/serega_db/{domain}/'
temp_sample = f'eeg_xai_code/serega_eval/result_image/{domain}/'

# Load synthetic data 
print(f'Loading {domain} eeg dataset with noise {snr}...')
all_data, all_gt, all_labels, channel_names = LoadSyntheticData(data_path, domain)
n_channels = len(channel_names)

# Data processing
print('Normalisation...')
data_norm = NormData(all_data)
data_norm, labels, data_gt = ShuffledData(data_norm, all_labels, all_gt)

# Generate random labels
random_labels = array(GetRandLabels (labels))

# Save database
print(f'Save dataset at {data_path}')
save(f'{data_path}/{snr}_{domain}_data.npy', data_norm)
save(f'{data_path}/{snr}_{domain}_gt.npy', data_gt)
save(f'{data_path}/{snr}_{domain}_labels.npy', labels)
save(f'{data_path}/{snr}_{domain}_random_labels.npy', random_labels)
save(f'{data_path}/{snr}_{domain}_channel_names.npy', channel_names)

# Show sample
rand_idx = random.randint(0, data_norm.shape[0]-1)
sample = data_norm[rand_idx]
lbl_sample = labels[rand_idx]
chan_num = data_norm.shape[1]

print (f'\nRandom sample index: {rand_idx}')
print(f'max : {max(sample)}')
print(f'min : {min(sample)}')
print('Plotting sample...')
plot_data = PlotData(sample, lbl_sample, chan_num=chan_num)
print('Ploting norm channels...')
plot_ch = PlotChannels(sample, channel_names, chan_num=chan_num)

# Save sample
plot_data.savefig(f"{temp_sample}{rand_idx}_sample_eeg.png")
plot_ch.savefig(f"{temp_sample}{rand_idx}_sample_ch.png")