"""
This script trains an EEG real dataset model.

1. Configuration and Argument Parsing
2. Data Loading
3. Label Encoding and Initialisation
4. Apply data pre-processing
4. Training and Validation
5. Computes average and standard deviation of performance metrics 
6. Save and display summary

"""

#Dependencies
from argparse import ArgumentParser
from numpy import array, min, max, random
from pandas import DataFrame, ExcelWriter
from torch import load, device, cuda
from torchsummary import summary
from time import time
from pandas import read_excel
from sklearn.model_selection import train_test_split
from warnings import filterwarnings
filterwarnings('ignore')

from sys import path
path.append('/home/ade/eeg_xai_code/utils')
from eeg_train import GetSubjectTrainData, GetDataLoader, TrainingModel, EvalModel, AdjustDimenson
from eeg_models import SelectModel
from eeg_dbf import LblEncoding,  WindowingEEg, PlotData, PlotChannels

import numpy as np
import torch


# Parameter setup
parser = ArgumentParser()
parser.add_argument('--db_name', type=str, required=False, choices=['predict', 'modma', 'mumtaz'], default= "modma")
parser.add_argument('--model_name', type=str, required=False, choices=['TempCNN','TwoDCNN', 'EEGNet', 'DeprNet', 'ResNet'], default="EEGNet")
args = parser.parse_args()
db_name = args.db_name
model_name = args.model_name
disp = False #sample visualisation

# Db info
if db_name == 'mumtaz':
    n_chan = 19
    sampling_rate = 256
elif db_name == 'modma':
    sampling_rate = 250
    n_chan = 128
elif db_name == 'predict':
    sampling_rate = 500
    n_chan = 64

# Provide link for data and result --Replace with suitable link
data_link = f'eeg_xai_code/real_db_eval/real_db/{db_name}/'
data_path = f'{data_link}np_eeg_{db_name}/'
info_path = f'{data_link}infodb_{db_name}.xlsx'
ch_path = f'{data_link}chNames_{db_name}.xlsx'
model_link = f'eeg_xai_code/real_db_eval/model/{db_name}/{model_name}/'
result_link = f'eeg_xai_code/real_db_eval/result/{db_name}/{model_name}/'
imgage_link = f'eeg_xai_code/real_db_eval/result_image/{db_name}/{model_name}/'

# Subject info 
info_label = read_excel(info_path)
channel_names = read_excel(ch_path)
channel_names = channel_names.values.tolist()

h_data = info_label.loc[info_label['status'] == 'Healthy', 'id'].values
d_data = info_label.loc[info_label['status'] == 'Depression', 'id'].values
print(f'Healty subjects: {h_data}')
print(f'Depressed subjects: {d_data}')

# Training param inisialisation
n_class = 2
n_sample = sampling_rate
n_window = 1000
n_exp = 5
n_epochs = 50
n_patience = 10
opt =  "Adam"  # choice :"SGD" "RMSprop" "Adagrad": "Adamax, Adam"

print('-------------------------------------------------------------------------')
print(f'Training {db_name} dataset using {model_name}')

# select model
my_device = device('cuda' if cuda.is_available() else 'cpu')
cuda.empty_cache()
train_model = SelectModel(model_name, n_class=n_class, n_chan=n_chan, n_samples =n_sample)
train_model.to(my_device)

print(f'\nModel structure:')
summary(train_model, (1, n_chan, n_sample))

# Dataset pre-processing
if db_name == 'predict' :
    rm_seg = 100
    batch_size = 32
    lr =  0.0001 #TempCNN : 0.01 >> not convergence
    abs_norm = False
    bandpass_fil = False
if db_name == 'modma' :
    rm_seg = 15
    batch_size = 128
    lr = 0.0001 #TempCNN, TwoDCNN= 0.1
    abs_norm = True
    bandpass_fil = True
elif db_name == 'mumtaz':
    rm_seg = 15
    batch_size = 128
    lr =  0.0001
    abs_norm = True
    bandpass_fil = True
notch_fil = False
fastICA_fil = False

# implement pre-processing
data_dict, label_dict = WindowingEEg(data_path, info_path, n_chan= n_chan, sampling_rate=sampling_rate, 
                                     len_window=n_sample, n_window=n_window, rm_seg = rm_seg, abs_norm=abs_norm, 
                                     bandpass_fil=bandpass_fil, notch_fil=notch_fil, fastICA_fil=fastICA_fil)

dict_keys = (list(data_dict.keys()))
print(f'Number of file extracted: {len(dict_keys)}')

#Sample visualisation

if disp:
    rand_obj = random.choice(dict_keys) 
    sample = data_dict[rand_obj]
    lbl_sample = label_dict[rand_obj]

    print (f'\nRandom sample index: {rand_obj}')
    print(f'Label: {lbl_sample[0]}')
    print(f'Data shape: {array(sample).shape}')
    print(f'max : {max(sample)}')
    print(f'min : {min(sample)}')

    print('Plotting sample...')
    plot_data = PlotData(sample[0], lbl_sample[0], chan_num=n_chan)
    plot_data.savefig(f"{imgage_link}{rand_obj}_sample_eeg.png")
    print('Ploting channels...')
    plot_ch = PlotChannels(sample[0], channel_names, chan_num=10)
    plot_ch.savefig(f"{imgage_link}{rand_obj}_sample_ch.png")

#searching for well data representation
for seed in range(1):
    if db_name == 'modma' :
        seed_id = 3 #random.choice(np.arange(1, 100))
    elif db_name == 'mumtaz' :
        seed_id =  67 
    elif db_name == 'predict' :
        seed_id = 78 #78 balanced data  #57 full data 

    print('***************************************************************************')
    print('Seed_id', seed_id)

    #spliting data subject-based
    h_trainval_id, h_test_id = train_test_split(h_data, test_size=0.2, random_state=seed_id)
    d_trainval_id, d_test_id = train_test_split(d_data, test_size=0.2, random_state=seed_id)
    h_train_id, h_val_id = train_test_split(h_trainval_id, test_size=0.125, random_state=seed_id)
    d_train_id, d_val_id = train_test_split(d_trainval_id, test_size=0.125, random_state=seed_id)

    subject_id = {}
    subject_id['train'] = np.concatenate((h_train_id, d_train_id))
    subject_id['val'] = np.concatenate((h_val_id, d_val_id))
    subject_id['test'] = np.concatenate((h_test_id, d_test_id))

    print(f"Subject size on train sets:{len(subject_id['train'])}")
    print(f"Subject on train sets:{(subject_id['train'])}")
    print(f"Subject size on validation sets:{len(subject_id['val'])}")
    print(f"Subject on validation sets:{(subject_id['val'])}")
    print(f"Subject size on test sets:{len(subject_id['test'])}")
    print(f"Subject on test sets:{(subject_id['test'])}")

    # Set best model paths
    best_model_path = f'{model_link}{db_name}_best_{model_name}_seed{seed_id}.pth'

    # Training model
    start_time = time()
    exp_history = []
    n_exp =5
        
    X_train, y_train, X_val, y_val, X_test, y_test = GetSubjectTrainData (data_dict, label_dict, subject_id)
    X_train, X_test, X_val = AdjustDimenson(X_train, X_test, X_val)

    y_train_encoded = LblEncoding(y_train, num_classes=n_class)
    y_test_encoded = LblEncoding(y_test, num_classes=n_class)
    y_val_encoded = LblEncoding(y_val, num_classes=n_class)

    # implementing training data 
    for exp_id in range (n_exp):
        print('=========================================================================')
        print(f"Exp-{exp_id}")
        
        # Define the model
        train_model = SelectModel(model_name, n_class=n_class, n_chan=n_chan, n_samples =n_sample)
        # Load data every fold

        best_fold_model_path = f'{model_link}{db_name}_best_exp{exp_id}_{model_name}_seed{seed_id}.pth'
        print(best_fold_model_path)


        train_loader, val_loader, test_loader = GetDataLoader(X_train, y_train_encoded, X_val, y_val_encoded, 
                                                            X_test, y_test_encoded, batch_size)
        
        # Start train
        loss_train_epoch, acc_train_loss, loss_val_epoch, acc_val_loss = TrainingModel (train_model, train_loader, 
                                                                                        val_loader, n_epochs, n_patience, 
                                                                                        opt, lr, best_model_path, best_fold_model_path)

        # Save best fold model
        best_fold_model = SelectModel(model_name, n_class=n_class, n_chan=n_chan, n_samples =n_sample)
        best_fold_model.load_state_dict(load(best_fold_model_path))

        # Eval best fold model
        eval_result, y_pred = EvalModel (best_fold_model, test_loader)
        eval_result =  [loss_train_epoch, acc_train_loss, loss_val_epoch, acc_val_loss] + eval_result
        exp_history.append(eval_result)

    end_time = time()
    execution_time = end_time - start_time

    # Write history ^_^
    exp_history = [[round(float(x), 6) for x in sublist] for sublist in exp_history]
    index =  [f'exp-{i}' for i in range(n_exp)]
    exp_history_df = DataFrame(exp_history, columns=['train_loss', 'train_acc', 'val_loss', 'val_acc',
                                                    'test_loss', 'test_acc', 'f1-score', 'precision' ,'recall' ], index=index)
    average_history = exp_history_df.mean()
    std_history = exp_history_df.std()
    exp_history_df.loc['avg'] = average_history
    exp_history_df.loc['std'] = std_history

    #Save training information
    info_path = f'{result_link}{db_name}_{model_name}_info_seed{seed_id}.xlsx'
        
    data_info = DataFrame(columns=['Variable', 'value'])
    data_info.loc[0] = ['Dataset', db_name]
    data_info.loc[1] = ['Num of file', len(dict_keys)]
    data_info.loc[2] = ['Sampling rate', sampling_rate]
    data_info.loc[3] = ['Sample size', n_sample]
    data_info.loc[4] = ['Channel', n_chan]
    data_info.loc[5] = ['Data path', data_path]
    data_info.loc[6] = ['Seed id', seed_id]
    data_info.loc[7] = ['Model', model_name]
    data_info.loc[8] = ['Fold', n_exp]
    data_info.loc[9] = ['Number of epochs', n_epochs]
    data_info.loc[10] = ['Patience', n_patience]
    data_info.loc[11] = ['Optimiser', opt]
    data_info.loc[12] = ['Learning rate', lr]
    data_info.loc[13] = ['LBatch size', batch_size]
    data_info.loc[14] = ['Absolute normalisation', abs_norm]
    data_info.loc[15] = ['Bandpass filtering', bandpass_fil]
    data_info.loc[16] = ['Notch filtering', notch_fil]
    data_info.loc[17] = ['FastICA ', fastICA_fil]
    data_info.loc[18] = ['Best model path',best_model_path ]
    data_info.loc[19] = ['Train subject', [subject_id['train']][0]]
    data_info.loc[20] = ['Val subject', [subject_id['val']][0]]
    data_info.loc[21] = ['Test subject', [subject_id['test']][0]]
    data_info.loc[22] = ['Execution time (s)', f'{int(execution_time)} s']

    with ExcelWriter(info_path, engine='openpyxl') as writer:
        data_info.to_excel(writer, sheet_name='data', index=False)
        exp_history_df.to_excel(writer, sheet_name='eval history', index=True)

    # Display result
    print(f'\nInfo: \n{data_info}')
    print(f'History: \n {exp_history_df}')
    print(f'\nResult of experimet on {db_name} dataset using {model_name} saved in: {info_path}')
    print(f'Seed: {seed_id} batch size: {batch_size}, optimizer: {opt}\n')
    print(f'Sampling:{n_sample}, lr:{lr}\n')