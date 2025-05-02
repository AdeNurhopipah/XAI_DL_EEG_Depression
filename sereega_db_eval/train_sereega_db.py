"""
This script trains an EEG classification model with specified configurations, performing k-fold cross-validation 
and saving results. Key sections include:

- Data Loading: Loads EEG data, ground-truth, and labels from `.npy` files based on configuration.
- Label Encoding and Initialization: Encodes labels, sets model parameters, and initializes paths.
- Training and Cross-Validation: Runs k-fold cross-validation:
   - For each fold, selects and trains the model, saves the best fold model, and evaluates on test data.
- Summary: Prints experiment details, model summary, and training metrics, and saves results to an Excel file.

Author: Ade Nurhopipah
Email: ade.nurhopipah@postgrad.otago.ac.nz
"""

#Dependencies
from argparse import ArgumentParser
from sys import path
from numpy import load, unique, array, swapaxes, expand_dims
from pandas import DataFrame, ExcelWriter
from torch import load as torch_load
from torchsummary import summary
from time import time
from warnings import filterwarnings
filterwarnings('ignore')

path.append('/home/ade/eeg_xai_code/utils')
from eeg_train import GetDataFold, GetDataLoader, TrainingModel, EvalModel, AdjustDimenson
from eeg_models import SelectModel
from eeg_dbf import LblEncoding


# Parameter setup
parser = ArgumentParser()
parser.add_argument('--domain', type=str, required=False, choices=['temporal', 'spectral', 'spatial'], default="temporal")
parser.add_argument('--model', type=str, required=False, choices=['TwoDCNN', 'EEGNet'], default="TwoDCNN")
parser.add_argument('--random_label', type=str, required=False, default="True")
args = parser.parse_args()
domain = args.domain
model_name = args.model
snr =  '-3.5'

if args.random_label == 'True':
    random_label = True 
else:
    random_label = False

# Link for data, model, and model evaluation result 
# Replace with the right path
data_link= f'eeg_xai_code/serega_eval/serega_db/{domain}/{snr}_{domain}' 
model_link = f'eeg_xai_code/serega_eval/model/{domain}/{model_name}/'
result_link = f'eeg_xai_code/serega_eval/result/{domain}/{model_name}/'

print('-------------------------------------------------------------------------')
print(f'Training {domain} eeg with {snr} noise using {model_name}')
print(f'Label randomised : {random_label }')


# Load numpy eeg data
try :
    print(f'Opening eeg data files from {data_link}...')
    eeg_data = load(f'{data_link}_data.npy') 
    labels = load(f'{data_link}_labels.npy' ) 
    channel_names = load(f'{data_link}_channel_names.npy')
except ValueError as e:
    print(f"Error in loading file: {e}")

n_class = len(unique(labels))
n_chan = array(eeg_data).shape[1]
n_sample = array(eeg_data).shape[2]
labels_encoded = LblEncoding (labels, n_class)

print(eeg_data.shape)

# Random setting and paths
if random_label :
    random_labels_path = f'{data_link}_random_labels.npy'
    random_labels = load(random_labels_path)
    random_labels_encoded = LblEncoding (random_labels, n_class)

    best_model_path = f'{model_link}{snr}_{domain}_best_random_label_{model_name}.pth'
    info_path = f'{result_link}{snr}_{domain}_{model_name}_info_random.xlsx'
else:
    best_model_path = f'{model_link}{snr}_{domain}_best_{model_name}.pth'
    info_path = f'{result_link}{snr}_{domain}_{model_name}_info.xlsx'

#Training parameters inisialisation
n_fold = 5 #n fold at least = 3 to accommodate train, val and eval fold
n_epochs = 30
n_patience = 10
lr = 0.0001 
best_val_loss = float('inf')
generator_size = len(eeg_data)//n_fold
opt =  "Adam"

# Training model
start_time = time()
fold_history = []
for fold_id in range (n_fold):
    print('=========================================================================')
    print(f"Fold-{fold_id}")
    
    # Define the model
    train_model = SelectModel(model_name, n_class=n_class, n_chan=n_chan, n_samples =n_sample)

    # Load data every fold
    if random_label :
        best_fold_model_path = f'{model_link}{snr}_{domain}_random_label_fold{fold_id}_{model_name}.pth'
        X_train, y_train, X_val, y_val, X_test, y_test = GetDataFold(eeg_data, random_labels_encoded, n_fold, fold_id, generator_size)
    else:
        best_fold_model_path = f'{model_link}{snr}_{domain}_best_fold{fold_id}_{model_name}.pth'
        X_train, y_train, X_val, y_val, X_test, y_test = GetDataFold(eeg_data, labels_encoded, n_fold, fold_id, generator_size)
    
    X_train, X_test, X_val = AdjustDimenson(X_train, X_test, X_val)
    
    batch_size =  256
    train_loader, val_loader, test_loader = GetDataLoader(X_train, y_train, X_val, y_val, X_test, y_test, batch_size)
    
    # Start train
    loss_train_epoch, acc_train_loss, loss_val_epoch, acc_val_loss = TrainingModel (train_model, train_loader, val_loader, 
                                                                                    n_epochs, n_patience,  opt, lr,
                                                                                    best_model_path, best_fold_model_path)
    # Save best fold model
    best_fold_model = SelectModel(model_name, n_class=n_class, n_chan=n_chan, n_samples =n_sample)
    best_fold_model.load_state_dict(torch_load(best_fold_model_path))

    # Eval best fold model
    eval_result, y_pred = EvalModel (best_fold_model, test_loader)
    eval_result =  [loss_train_epoch, acc_train_loss, loss_val_epoch, acc_val_loss] + eval_result
    fold_history.append(eval_result)

end_time = time()
execution_time = end_time - start_time

# Write model history 
fold_history = [[round(float(x), 6) for x in sublist] for sublist in fold_history]
index =  [f'Fold-{i+1}' for i in range(n_fold)]
fold_history_df = DataFrame(fold_history, columns=['train_loss', 'train_acc', 'val_loss', 'val_acc','test_loss', 
                                                   'test_acc', 'f1-score', 'precision' ,'recall' ], index=index)
average_history = fold_history_df.mean()
std_history = fold_history_df.std()
fold_history_df.loc['avg'] = average_history
fold_history_df.loc['std'] = std_history

#Save training information   
data_info = DataFrame(columns=['Variable', 'value'])
data_info.loc [0] = ['Domain', domain]
data_info.loc[1] = ['Data size', eeg_data.shape]
data_info.loc[2] = ['Data path', data_link]
data_info.loc[3] = ['SNR', snr]
data_info.loc[4] = ['Model', model_name]
data_info.loc[5] = ['Fold', n_fold]
data_info.loc[6] = ['Number of epochs', n_epochs]
data_info.loc[7] = ['Patience', n_patience]
data_info.loc[8] = ['Best model path',best_model_path ]
data_info.loc[9] = ['Execution time (s)', f'{int(execution_time)} s']

with ExcelWriter(info_path, engine='openpyxl') as writer:
    data_info.to_excel(writer, sheet_name='data', index=False)
    fold_history_df.to_excel(writer, sheet_name='eval history', index=True)

# Show model summary
print(f'\nModel summary:')
summary(train_model, (1, n_chan, n_sample))

# Display result
print(f'\nInfo: \n{data_info}')
print(f'History: \n {fold_history_df}')
print(f'\nExperiment on {domain} domain using {model_name} are saved in: {info_path}\n')
