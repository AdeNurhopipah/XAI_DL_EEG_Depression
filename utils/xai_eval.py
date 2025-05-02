'''
These module facilitate XAI analysis, model interpretability, and evaluation of sensitivity and robustness for EEG-based models.

Sensitivity and Robustness Analysis:
- GetSensitivity: Computes sensitivity scores for model evaluation.
- EvalSensitivity: Applies XAI methods to test sensitivity on a dataset.
- GetRobustness: Calculates robustness scores for model assessment.
- EvalRobustness: Implements XAI methods to evaluate robustness on a dataset.

Explainability Methods:
- GetExplainer: Retrieves an XAI method explainer.
- GetConvLayers: Extracts convolutional layers from a model for explainability purposes.
- DefineLRPRules: Sets additional Layer-wise Relevance Propagation (LRP) rules.

Normalisation and Thresholding:
- MaxAbsNorm: Performs max absolute normalisation for data scaling.
- Thresholding: Applies thresholding based on a given value.
- QuartileThresholding: Thresholds data using quartile-based criteria.
- GetTaperredGt: Generates tapered ground truth data.

Visualisation:
- PlotXai: Visualizes XAI results for interpretability.
- Top3Channel: Identifies and extracts the top 3 most important EEG channels based on importance scores.
- TopoplotChan: Creates a topographical brain map using averaged XAI scores.
- BarplotChan: Creates a bar plot using averaged XAI scores.

Author: Ade Nurhopipah
Email: ade.nurhopipah@postgrad.otago.ac.nz
'''

#dependencies
from numpy import arange, expand_dims, array, where, zeros_like, all, mean
from numpy import max, abs, quantile, min
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from pandas import DataFrame
from scipy import signal
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from torch import device, cuda, tensor, float32, nn, Tensor, randn
from mne import create_info, EvokedArray, viz
from captum.attr import Saliency, IntegratedGradients, DeepLift, LRP
from captum.attr import Deconvolution, GuidedBackprop, InputXGradient, GuidedGradCam
from captum.attr._utils.lrp_rules import EpsilonRule
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, FullGrad, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sys import path

path.append('/home/ade/eeg_xai_code/utils')
from eval_methods import EvalScore


#-----------------------------------------------------------------------------------------------------------------------#
# SENSITIVITY  AND ROBUSTNESS TEST

# Calculate sensitivity scores
def GetSenitivity (model, X_test, labels, preds, gt_test, th_gt, th_xai_list, xai_methods_list = ['Saliency'], eval_method='RMA', index=None):
    scores_dict ={}
    all_scores = DataFrame(columns=xai_methods_list)
    if index == None:
        index = arange(len(X_test))

    #temporal 5-2-5
    for xai_method in xai_methods_list: 
        if xai_method in ['Saliency', 'Deconv','Guid-BP', 'DeepLift','IxG','IG','Guid-GCam', 'LRP']:
            th_xai = th_xai_list[0]
        elif xai_method in [ 'GradCam','GradCam++', 'ScoreCam','FullGrad','LayerCAM']: 
            th_xai = th_xai_list[1]

        print(f'Implementing {xai_method}...')
        eval_scores, score_dict= EvalSensitivity(model, X_test, labels, preds, gt_test, th_gt, th_xai,  
                          xai_method, eval_method, index)

        all_scores[xai_method] = eval_scores
        scores_dict[xai_method] = score_dict
             
    all_scores = DataFrame(all_scores)
    all_scores.index = index

    return all_scores, scores_dict

# Implement XAI methods for sensitivity test on a dataset
def EvalSensitivity(model, X_test, labels, preds, gt_test, th_gt, th_xai,  
                          xai_method, eval_method, index):
    
    my_device = device("cuda" if cuda.is_available() else "cpu")
    model = model.to(my_device)
    scaler = MinMaxScaler() #MinMaxScaler() # StandardScaler, MaxAbsScaler

    selected_layers=[]
    if xai_method in ['Guid-GCam', 'GradCam','GradCam++', 'ScoreCam', 'FullGrad','LayerCAM']:  
       selected_layers = GetConvLayers(model)
        
    if xai_method != 'LRP':
        explainer = GetExplainer(model, xai_method, selected_layers) 
   
    xai_score = []
    eval_scores = []
    for idx in range(len(X_test)): 
        #load data
        X_sample = X_test[idx]
        X_sample = expand_dims(X_sample, axis=0)
        X_sample = tensor(X_sample, dtype=float32, requires_grad=True).to(my_device)
        y_sample = int(labels[idx])
        pred_sample = int(preds[idx])
        gt_sample = gt_test[idx]

        #if (idx % 100) == 0:
        print(f'{xai_method}-Sample {index[idx]}, label : {y_sample}, prediction: {pred_sample}...')
           
        if xai_method in ['Saliency', 'Deconv','Guid-BP', 'DeepLift','IxG','IG']:
            important_map = explainer.attribute(X_sample, target=pred_sample)
        elif xai_method =='Guid-GCam':
            important_map = explainer.attribute(X_sample, target=pred_sample, interpolate_mode="bilinear")
        elif xai_method in [ 'GradCam','GradCam++', 'ScoreCam','FullGrad','LayerCAM']: 
            target_class = [ClassifierOutputTarget(pred_sample)]
            important_map = explainer(input_tensor=X_sample, targets=target_class)
        elif xai_method== 'LRP':
            model = DefineLRPRules(model)
            explainer = LRP(model)
            important_map = explainer.attribute(X_sample, target=pred_sample)
        else :
            print(f'Method {xai_method} not defined')
        
        # Ensure `important_map` is converted to NumPy
        if isinstance(important_map, Tensor):
            important_map = important_map.squeeze().cpu().detach().numpy()
        else:
            important_map = important_map.squeeze()  
        
        # Normalisation and thersholding
        if eval_method == 'RMA':
            gt_th = GetTaperredGt(gt_sample, th_gt)
            important_map = MaxAbsNorm(important_map)
            important_map = Thresholding(important_map, th_xai, binary =True) 
        elif eval_method == 'CS': 
            gt_th = scaler.fit_transform(gt_sample.T).T
            important_map = scaler.fit_transform(important_map.T).T
                 
        eval_score = EvalScore (important_map, gt_th, method = eval_method)

        #Extract sample data for visualisation
        eval_scores.append(eval_score)
        if idx< 5:
            xai_score.append(important_map)

    return eval_scores, xai_score


# Calculate robustness scores
def GetRobustness (ori_model, rand_model, X_test, labels, rand_labels, ori_preds, rand_preds,  
                          xai_methods_list = ['Saliency'], eval_method='PC', index=None):
    
    if index == None:
        index = arange(len(X_test))

    ori_dict, rand_dict ={}, {}
    all_scores = DataFrame(columns=xai_methods_list)
    for xai_method in xai_methods_list: 
        print(f'Implementing {xai_method}...')
        eval_scores, ori_score, rand_score = EvalRobustness (ori_model, rand_model, X_test, labels, rand_labels,
                                                             ori_preds, rand_preds, xai_method, eval_method, index)
        
        all_scores[xai_method] = eval_scores
        ori_dict[xai_method] = ori_score
        rand_dict[xai_method] = rand_score

    all_scores = DataFrame(all_scores)
    all_scores.index = index

    return all_scores, ori_dict, rand_dict


#Apply XAI methods for robustness test on a dataset
def EvalRobustness(ori_model, rand_model, X_test, labels, rand_labels, ori_preds, rand_preds, 
                   xai_method, eval_method, index):
    
    my_device = device("cuda" if cuda.is_available() else "cpu")

    ori_sel_layers, rand_sel_layers = [], []
    if xai_method in ['Guid-GCam', 'GradCam','GradCam++', 'ScoreCam', 'FullGrad','LayerCAM']:  
        ori_sel_layers = GetConvLayers(ori_model)
        rand_sel_layers = GetConvLayers(rand_model)

    if xai_method != 'LRP':
        ori_explainer = GetExplainer(ori_model, xai_method, ori_sel_layers)
        rand_explainer = GetExplainer(rand_model, xai_method, rand_sel_layers)

    ori_score, rand_score= [], []
    eval_scores = []

    for idx in range(len(X_test)):
        #load data
        X_sample = X_test[idx]
        X_sample = expand_dims(X_sample, axis=0)
        X_sample = tensor(X_sample, dtype=float32, requires_grad=True).to(my_device)
        y_sample = int(labels[idx])
        y_rand = int(rand_labels[idx]) 
        ori_pred = int(ori_preds[idx])
        rand_pred = int(rand_preds[idx])

        #if (idx % 100) == 0:
        print(f'{xai_method}-Sample {index[idx]}, label : {y_sample}, pred: {ori_pred}, random label: {y_rand}, random pred: {rand_pred}...')

        if xai_method in ['Saliency', 'Deconv','Guid-BP', 'DeepLift','IxG','IG']:
            ori_map = ori_explainer.attribute(X_sample, target=ori_pred)
            rand_map = rand_explainer.attribute(X_sample, target=rand_pred)
        elif xai_method =='Guid-GCam':
            ori_map = ori_explainer.attribute(X_sample, target=ori_pred, interpolate_mode="bilinear")
            rand_map = rand_explainer.attribute(X_sample, target=rand_pred, interpolate_mode="bilinear")
        elif xai_method in [ 'GradCam','GradCam++', 'ScoreCam','FullGrad','LayerCAM']: 
            ori_class = [ClassifierOutputTarget(ori_pred)]
            rand_class = [ClassifierOutputTarget(rand_pred)]
            ori_map = ori_explainer(input_tensor=X_sample, targets=ori_class)
            rand_map = rand_explainer(input_tensor=X_sample, targets=rand_class)
        elif xai_method== 'LRP':
            ori_model = DefineLRPRules(ori_model)
            ori_explainer = LRP(ori_model)
            ori_map = ori_explainer.attribute(X_sample, target=ori_pred)
            rand_model = DefineLRPRules(rand_model)
            rand_explainer = LRP(rand_model)
            rand_map = rand_explainer.attribute(X_sample, target=rand_pred)
        else :
            print(f'Method {xai_method} not defined')
        
        if isinstance(ori_map, Tensor):
            ori_map = ori_map.squeeze().cpu().detach().numpy()
            rand_map = rand_map.squeeze().cpu().detach().numpy()
        else:
            ori_map = ori_map.squeeze() 
            rand_map = rand_map.squeeze()  

        #print(min(rand_map), max(rand_map))
        if eval_method == 'PC':
            ori_map = MaxAbsNorm(ori_map)
            rand_map = MaxAbsNorm(rand_map)
        elif eval_method =='SSIM':
            ori_map = abs(ori_map)
            rand_map = abs(rand_map)

        eval_score = EvalScore (ori_map, rand_map, method = eval_method)
        eval_scores.append(eval_score)

        if idx< 5:
            ori_score.append(ori_map)
            rand_score.append(rand_map)

    return eval_scores, ori_score, rand_score


#-----------------------------------------------------------------------------------------------------------------------#
# XAI METHODS

# Get XAI method explainer
def GetExplainer (model, xai_method, selected_layers):
    if xai_method == 'Saliency':
        explainer = Saliency(model)
    elif xai_method == 'Deconv':
        explainer = Deconvolution(model)
    elif xai_method == 'Guid-BP':
        explainer = GuidedBackprop(model)
    elif xai_method == 'DeepLift':
        explainer = DeepLift(model)   
    elif xai_method == 'IxG': 
        explainer = InputXGradient(model)
    elif xai_method == 'IG':
        explainer = IntegratedGradients(model)

    else:  
        if len(selected_layers)==0:
            print('No conv layer passed!')
        else:
            #GradCam Family
            if xai_method =='Guid-GCam':
                explainer = GuidedGradCam(model, selected_layers[-1]) 
            elif xai_method == 'GradCam':
                explainer = GradCAM(model=model, target_layers=[selected_layers[-1]]) 
            elif xai_method == 'GradCam++':
                explainer = GradCAMPlusPlus(model=model, target_layers=[selected_layers[-1]]) 
            elif xai_method == 'ScoreCam':
                explainer = ScoreCAM(model=model, target_layers=[selected_layers[-1]]) 
            elif xai_method == 'FullGrad':
                #target_layers is ignored in FullGrad. All bias layers will be used instead
                explainer = FullGrad(model=model,target_layers= selected_layers) 
            elif xai_method == 'LayerCAM':
                #extracted all layer
                explainer = LayerCAM(model=model, target_layers=selected_layers) 
            else:
                print(f'Method {xai_method} not found!')

    return explainer


# Extract convolutional layers on a model
def GetConvLayers(model):
    selected_layers = []
    for layer in model.modules():
        if isinstance(layer, (nn.Conv1d, nn.Conv2d)):
            selected_layers.append(layer)
    if not selected_layers:
        raise ValueError("No Conv1d or Conv2d layer found in the model.")
    #print(selected_layers)

    return selected_layers

# Define rules on LRP
def DefineLRPRules(model):
    for layer in model.modules():
        if isinstance(layer, (nn.Flatten)):
            layer.rule = EpsilonRule()
    return model


#-----------------------------------------------------------------------------------------------------------------------#
# NORMALISATION AND THRESHOLDING

# Max absolute normalisation
def MaxAbsNorm(np_data):
    max_val = max(abs(np_data))
    if max_val == 0:
        print('Warning the data has maximum value=0. Returning zero numpy')
        return zeros_like(np_data)
    else:
      np_data = np_data / max_val

    return array(np_data)

# Tresholding based given value
def Thresholding(data, th, binary = True):
  if binary:
    return where((data > th) | (data < -th), 1, 0)
  else:
    return where((data > th) | (data < -th), data, 0)


# Thresholding based on quartile data
def QuartileThresholding (data, th, binary = True):
  if all(data == 0):  
    print("Warning: The input array is all zeros. Returning the same array.")
    return data
  
  data = abs (data)
  threshold = quantile(data, th)
  if binary:
    data = where(data >= threshold, 1, 0)
  else:
    data = where(data >= threshold, data, 0)
  return data

# Get tapered ground truth data
def GetTaperredGt (gt, th):
  tap_gt = zeros_like(gt)
  sample = abs(gt)
  sample_norm = MaxAbsNorm(sample)
  sample_th = where((sample_norm > th), 1, 0)

  for i in range (len(sample_th)):
    window = zeros_like(sample_th[i])
    indices = where(sample_th[i] == 1)[0]
    if len(indices) > 0:
        first_idx= indices[0]
        last_idx = indices[-1]
        window_tukey = signal.windows.tukey(len(window[first_idx:last_idx+1]),0)
        window[first_idx:last_idx+1] =window_tukey
        tap_gt[i] = window

  return tap_gt



#-----------------------------------------------------------------------------------------------------------------------#
# VISUALISATION

# Plot XAI method's results 
def PlotXai(dict_scores, sample_idx):
    list_data = list(dict_scores.values())
    list_desc = list(dict_scores.keys())
    
    # Define the number of subfigures per row
    max_plots_per_row = 5
    n_rows = (len(list_data) + max_plots_per_row - 1) // max_plots_per_row  
    fig = plt.figure(figsize=(20, 4 * n_rows))  
    gs = gridspec.GridSpec(n_rows, max_plots_per_row, figure=fig)
    
    # Loop over the methods and create subplots
    for i, (method, score) in enumerate(zip(list_desc, list_data)):
        row = i // max_plots_per_row  
        col = i % max_plots_per_row 
        
        ax = fig.add_subplot(gs[row, col])
        cax = ax.imshow(score, aspect='auto', cmap='viridis')
        fig.colorbar(cax, ax=ax, label='Amplitude')
        ax.set_title(method)
        ax.set_xlabel('Time Points')
        ax.set_ylabel('Channels')

    plt.suptitle(f"XAI methods result on sample {sample_idx}")
    plt.tight_layout()
    plt.show()
    
    return fig

# Extract top 3 channel based important matrix
def TopChannels (np_important_scores, ch_names, n_chans):
    averaged_scores = mean(np_important_scores, axis=1)
    max_values = sorted(enumerate(abs(averaged_scores)), key=lambda x: x[1], reverse=True)[:n_chans]
    indices, _ = zip(*max_values)
    chan_name, ch_value =[], []
    for i in indices:
        chan_name.append(ch_names[i])
        ch_value.append(averaged_scores[i])

    print("Top 3 Values:", ch_value)
    print("Indices:", chan_name)

    return ch_value, chan_name


# Create a topology brain map based averaged XAI score
def TopoplotChan(np_important_scores, ch_names, mag, montage, mark):
    # Create MNE Info object
    
    info = create_info(ch_names=ch_names, sfreq=1000, ch_types='eeg')
    info.set_montage(montage)
    
    # Create an EvokedArray object
    print(np_important_scores.shape)
    averaged_scores = mean(np_important_scores, axis=1)
    evoked = EvokedArray(np_important_scores, info)
    
    # Create subplots: topomap on the left, bar plot on the right
    fig, ax = plt.subplots(figsize=(8, 6))
    if mag:
        cmap = LinearSegmentedColormap.from_list('custom_cmap', ['blue', 'white', 'red'], N=256)
    else:
        cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', 'red'], N=256)

    if mark:
        names = ch_names
    else:
        names = None
    # Plot the topomap on the left subplot
    viz.plot_topomap(
        averaged_scores,
        pos=evoked.info,  
        cmap=cmap,
        show=False,  
        names= names,
        outlines='head',
        extrapolate = 'head',  
        sensors=True, 
        axes=ax
    )
    
    plt.show()
    return fig

# Create bar plot based averaged XAI score
def BarplotChan(np_important_scores, ch_names):
    # Averaging the scores across the XAI results
    averaged_scores = mean(np_important_scores, axis=1)

    # Plot the bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(ch_names[:len(np_important_scores)], averaged_scores, color='steelblue')
    ax.set_xlabel('Channels')
    ax.set_ylabel('Average Importance')
    ax.tick_params(axis='x', rotation=90)
    
    plt.tight_layout()
    plt.show()
    return fig



