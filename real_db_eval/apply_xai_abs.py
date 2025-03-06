"""


"""

#dependencies
from argparse import ArgumentParser
from torch import load as torch_load, device, cuda, max, no_grad, argmax
from torch import tensor, float32, Tensor
from torchsummary import summary
from time import time
from numpy import mean, expand_dims,swapaxes, array, random
from pandas import DataFrame, ExcelWriter, read_excel
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from captum.attr import LRP
from matplotlib import pyplot as plt
from warnings import filterwarnings
from mne import channels
from numpy import save
filterwarnings('ignore')

from sys import path
path.append('/home/ade/eeg_xai_code/utils')
from eeg_train import DataGenerator, DataLoader, HeatmapCM, EvalModel, GetSubjectData 
from eeg_dbf import LblEncoding, MaxAbsNorm, WindowingEEg
from xai_eval import GetConvLayers, GetExplainer, DefineLRPRules, TopoplotChan, TopChannels, BarplotChan
from eeg_models import SelectModel
 
# File arguments
parser = ArgumentParser()
parser.add_argument('--db_name', type=str, required=False,choices=['modma', 'predict', 'mumtaz'], default="modma")
parser.add_argument('--model_name', type=str, required=False, choices=['EEGNet'], default="EEGNet")
parser.add_argument('--xai_method', type=str, required=False, 
                    choices= ['Saliency', 'Deconv', 'Guid-BP' ,'Guid-GCam', 'DeepLift', 'LRP',
                    'IxG', 'GradCam', 'GradCam++', 'ScoreCam','FullGrad', 'LayerCAM', 'IG'], default ='DeepLift')
args = parser.parse_args()
db_name = args.db_name
model_name = args.model_name
xai_method = args.xai_method
my_device = device('cuda' if cuda.is_available() else 'cpu')

# Dataset info, pick best seed saved
if db_name == 'mumtaz':
    sampling_rate = 256
    n_chan = 19
    seed_id = 67 #all =51
elif db_name == 'modma':
    sampling_rate = 250
    n_chan = 128
    seed_id = 3
elif db_name == 'predict':
    sampling_rate = 500
    n_chan = 64
    seed_id = 78 
    
n_class = 2
n_sample = sampling_rate
n_window = None
test_subj = 'all' #'all' #'test' >> implement to all subject or test dataset only 

# Provide links >> Replace with suitable link
data_link = f'eeg_xai_code/real_db_eval/real_db/{db_name}/'
data_path = f'{data_link}np_eeg_{db_name}/'
info_path = f'{data_link}infodb_{db_name}.xlsx'
ch_path = f'{data_link}chNames_{db_name}.xlsx'

info_link = f'eeg_xai_code/real_db_eval/'
result_link = f'{info_link}xai_result/{db_name}/{model_name}/{xai_method}/'
spliting_path = f'{info_link}result/{db_name}/{model_name}/{db_name}_{model_name}_info_seed{seed_id}.xlsx'
model_eval_path = f'{info_link}model/{db_name}/{model_name}/{db_name}_best_{model_name}_seed{seed_id}.pth'
result_path =  f'{result_link}{db_name}_{xai_method}_{model_name}_{test_subj}.xlsx'

print('-------------------------------------------------------------------------')
print(f'Training {db_name} dataset using {model_name}')

#Select model saved
my_device = device('cuda' if cuda.is_available() else 'cpu')
train_model = SelectModel(model_name, n_class=n_class, n_chan=n_chan, n_samples =n_sample)
train_model.to(my_device)
print(f'\nModel structure:')
summary(train_model, (1, n_chan, n_sample))

# Dataset pre-processing
if db_name == 'predict' :
    rm_seg = 100 #cut noise part in the end 
    batch_size =  32
    abs_norm = False
    bandpass_fil = False
else :
    rm_seg = 15
    batch_size =  128
    abs_norm = True
    bandpass_fil = True
notch_fil = False
fastICA_fil = False

info_label = read_excel(info_path)
channel_names = read_excel(ch_path, header= None)
channel_names = channel_names.values.tolist()
channel_names = [item[0] for item in channel_names]

# Pick subjects 
spliting_df = read_excel(spliting_path)
if test_subj =='all':  
    test_id = info_label['id'] 
elif test_subj == 'test':
    print(spliting_df)
    test_id = spliting_df.loc[spliting_df['Variable'] == 'Test subject', 'value'].values[0]
    print(test_id)
    test_id = [item.strip() for item in test_id.strip("[]").split("'") if item.strip() and item.strip() != " "]
print(f'Data test:{test_id}')

h_data = info_label.loc[info_label['status'] == 'Healthy', 'id'].values
d_data = info_label.loc[info_label['status'] == 'Depression', 'id'].values

# Pre processing data
data_dict, label_dict = WindowingEEg(data_path, info_path, n_chan= n_chan, sampling_rate=sampling_rate, len_window=n_sample, 
                                     n_window=n_window, rm_seg=rm_seg, abs_norm=abs_norm, bandpass_fil=bandpass_fil, 
                                     notch_fil=notch_fil, fastICA_fil=fastICA_fil)
dict_keys = (list(data_dict.keys()))
print(f'Number of file extracted: {len(dict_keys)}')

#Set topomap layout
if db_name == 'modma':
    montage = channels.make_standard_montage('GSN-HydroCel-128')
    channel_names = montage.ch_names
elif db_name == 'mumtaz' :
    montage = channels.make_standard_montage('standard_1020')  
    channel_names = channel_names
elif db_name  == 'predict':
    montage = channels.make_standard_montage('biosemi64')
    channel_names = montage.ch_names
mag =True # score direction 
mark = True # display channel name on topomaps
display = True # save topomaps


print('=========================================================================')
h_avg_group, d_avg_group = [], []
num_all_obj, num_all_d, num_all_h = 0,0, 0
#all_result = DataFrame (columns = ['Total', 'TP', 'TN', 'Acc', 'Top3-val_h', 'Top3-ch_h', 'Top3-val_d', 'Top3-ch_d'])
test_obj_result = DataFrame(columns=['subject',  'accuracy', 'status', 'prediction', 'top3 ch', 'top3 val'])

#Load model
print(f'Link model: {model_eval_path}') 
eval_model = SelectModel(model_name, n_class=n_class, n_chan=n_chan, n_samples =n_sample)
eval_model.load_state_dict(torch_load(model_eval_path))
eval_model.to(my_device)
eval_model.eval()

count_id, num_d, num_h =0, 0, 0
h_avg, d_avg, all_avg = [], [], []
status_list, pred_list = [], []

#Implement XAI methods
print('\nImplement XAI methods...')
print(f'Method: {xai_method}')
for subject_id in test_id : 

    #Generate data per subject
    print(f'Subject: {subject_id}')
    X_test, y_test = GetSubjectData (data_dict, label_dict, [subject_id])
    y_test_encoded = LblEncoding(array(y_test), num_classes=n_class)
    status = info_label.loc[info_label['id'] == subject_id, 'status'].values[0]
    if status =='Healthy':
        status_list.append(0)
    else:
        status_list.append(1)

    X_test = expand_dims(X_test, axis=1)

    data_test = DataGenerator(X_test, y_test_encoded)
    test_loader = DataLoader(data_test, batch_size=8, shuffle=False)

    # Prediction stage
    eval_model.eval()
    eval_result, preds_obj = EvalModel (eval_model, test_loader, disp=False)

    pred_status = 'None'
    if eval_result[1]> 0.5:
        pred_status = status
    else:
        if status =='Healthy':
            pred_status = 'Depression'
        else:
            pred_status = 'Healthy'
    
    if pred_status =='Healthy':
        pred_list.append(0)
    else:
        pred_list.append(1)

    print(f'Status: {status}')
    print(f'Prediction: {pred_status}, with confidence {eval_result[1]}')

    # Only extract important score if acc > 50 %
    if eval_result[1]> 0.5:
        print(f'Implement {xai_method} on Subject: {subject_id}, Label: {status}...')

        xai_dict ={}
        sel_layers= []
        if xai_method in ['Guid-GCam', 'GradCam','GradCam++', 'ScoreCam', 'FullGrad','LayerCAM']:  
            sel_layers = GetConvLayers(eval_model)

        if xai_method != 'LRP':
            explainer = GetExplainer(eval_model, xai_method, sel_layers)

        xai_score = []
        for idx in range(len(X_test)): 
            #load data
            X_sample = X_test[idx]
            X_sample = expand_dims(X_sample, axis=0)
            X_sample = tensor(X_sample, dtype=float32, requires_grad=True).to(my_device)
            y_sample = int(y_test[idx])
            pred_sample = int(preds_obj[idx])

            if xai_method in ['Saliency', 'Deconv','Guid-BP', 'DeepLift','IxG','IG', 'k_Shap']:
                important_map = explainer.attribute(X_sample, target=pred_sample)
            elif xai_method =='Guid-GCam':
                important_map = explainer.attribute(X_sample, target=pred_sample, interpolate_mode="bilinear")
            elif xai_method in [ 'GradCam','GradCam++', 'ScoreCam','FullGrad','LayerCAM']: 
                target_class = [ClassifierOutputTarget(pred_sample)]
                important_map = explainer(input_tensor=X_sample, targets=target_class)
            elif xai_method== 'LRP':
                model = DefineLRPRules(eval_model)
                explainer = LRP(eval_model)
                important_map = explainer.attribute(X_sample, target=pred_sample)
            else :
                print(f'Method {xai_method} not defined')
            
            if isinstance(important_map, Tensor):
                important_map = important_map.squeeze().cpu().detach().numpy()
            else:
                important_map = important_map.squeeze()  

            important_map = MaxAbsNorm(important_map)
            xai_score.append(abs(important_map))
        
        # Average across epochs
        average_map = mean(xai_score, axis=0)
        xai_dict[xai_method] = average_map

        if status =='Healthy':
            h_avg.append(abs(average_map))
            num_h +=1
        else :
            d_avg.append(abs(average_map))
            num_d +=1 

        all_avg.append(abs(average_map))

        # Find top channels per subject
        values, ch_names = TopChannels (average_map, channel_names[:n_chan], 5)        

    else :
        print(f'Subject {subject_id} has a low accuracy. {xai_method} not implemented!')
        values, ch_names = '-', '-'
    
    test_obj_result.loc[count_id] = [subject_id, eval_result[1], status, pred_status,ch_names, values ]
    count_id += 1

# Average accross subjects
h_avg_group= mean(h_avg, axis=0)
d_avg_group= mean(d_avg, axis=0)
all_avg_obj = mean(all_avg, axis=0)

# Find top channels per group
h_values, h_ch_names = TopChannels (h_avg_group, channel_names[:n_chan], 5)
d_values, d_ch_names = TopChannels (d_avg_group, channel_names[:n_chan], 5)
all_values, all_ch_names = TopChannels (all_avg_obj, channel_names[:n_chan], 5) 

# Save average important scores
save(f"{result_link}h_avg_group_{db_name}_{xai_method}_{test_subj}.npy", h_avg_group)
save(f"{result_link}d_avg_group_{db_name}_{xai_method}_{test_subj}.npy", d_avg_group)
save(f"{result_link}avg_group_{db_name}_{xai_method}_{test_subj}.npy", all_avg_obj)

# Calculate prediction overall
true_pred = (test_obj_result['status'] == test_obj_result['prediction']).sum()
acc_fold = true_pred/len(test_obj_result)
num_all_obj = num_all_obj+ len(test_obj_result)
num_all_d = num_all_d + num_d
num_all_h = num_all_h + num_h

# Save result
test_obj_result.loc[count_id+2] = ['True positive:', num_d, '','', '', '']
test_obj_result.loc[count_id+3] = ['True pred:', true_pred, '','', '', '']
test_obj_result.loc[count_id+4] = ['True negative:', num_h, '', 'Healthy class ch:', h_ch_names , h_values]
test_obj_result.loc[count_id+5] = ['Total Acc:', acc_fold , '', 'Depression class ch:', d_ch_names , d_values]
test_obj_result.loc[count_id+6] = ['', '' , '', 'All ch:', all_ch_names , all_values]

with ExcelWriter(result_path, engine='openpyxl') as writer:
    test_obj_result.to_excel(writer, sheet_name=f'result', index=True)

# Save topomaps
if display :
    vis_h_avg_topo= TopoplotChan(h_avg_group, channel_names[:n_chan], mag=mag, montage=montage, mark=mark)
    vis_h_avg_topo.savefig(f"{result_link}avg_h_topomap_{test_subj}_{xai_method}_mark{mark}.png")
    vis_d_avg_topo= TopoplotChan(d_avg_group, channel_names[:n_chan], mag=mag, montage=montage, mark=mark)
    vis_d_avg_topo.savefig(f"{result_link}avg_d_topomap_{test_subj}_{xai_method}_mark{mark}.png")
    vis_avg_topo= TopoplotChan(all_avg_obj, channel_names[:n_chan], mag=mag, montage=montage, mark=mark)
    vis_avg_topo.savefig(f"{result_link}avg_topomap_{test_subj}_{xai_method}_mark{mark}.png")

    vis_h_avg_bar = BarplotChan(h_avg_group, channel_names[:n_chan])
    vis_h_avg_bar.savefig(f"{result_link}avg_h_barplot_{test_subj}.png")
    vis_d_avg_bar = BarplotChan(d_avg_group, channel_names[:n_chan])
    vis_d_avg_bar.savefig(f"{result_link}avg_d_barplot_{test_subj}.png")
    vis_avg_bar = BarplotChan(all_avg_obj, channel_names[:n_chan])
    vis_avg_bar.savefig(f"{result_link}avg_barplot_{test_subj}.png")

    heatmap_img = HeatmapCM(status_list, pred_list, f'Confusion matrix of {model_name})')   
    heatmap_img.savefig(f"{result_link}heatmap_{model_name}.png")

    print(f'\nImages have been saved at {result_link}')

print('\n--------------------------------------------------------------------------------------------------------')