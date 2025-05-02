'''
Functions for data handling, training and evaluation, and model and result visualisation

Data Handling:
- DataGenerator: Custom Dataset for loading data (X) and labels (y) with optional transformations.
- GetDataFold: Creates fold-specific training, validation, and test datasets for synthetic data.
- GetDataTest: Prepares test data for a specific fold for synthetic data.
- GetSubjectTrainData: Prepares training-ready input data from subject splits using:
    - GetSubjectData: Creates windowed data for specific subject IDs.
- GetDataLoader: Creates data loaders for training, validation, and testing.
- AdjustDimenson: Adjusting dimension of data for imodel input

Training and Evaluation:
- EarlyStoppingTrain: Implements early stopping and saves the best model based on validation loss.
- GetOptimizer: Configures the optimizer for training.
- TrainingModel: Trains the model with loss/accuracy tracking and early stopping.
- EvalModel: Evaluates performance metrics including accuracy, F1-score, precision, and recall.

Visualization and Analysis:
- HeatmapCM: Visualizes the confusion matrix as a heatmap (true vs. predicted labels).
- GetModelWeights: Extracts and inspects model weights.
- VisualizeWeights: Displays model weights as heatmaps.
- RandomiseWeights: Randomizes model weights for robustness testing.

Author: Ade Nurhopipah
Email: ade.nurhopipah@postgrad.otago.ac.nz

'''

# Import dependencies
import numpy as np
from matplotlib import pyplot as plt
from seaborn import heatmap
from copy import deepcopy
from torch import tensor, float32, device, cuda, save, no_grad, max, argmax, optim, nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score

#-----------------------------------------------------------------------------#
# DATA HANDLING
# Data generator class
class DataGenerator(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = tensor(X, dtype= float32)
        self.y = tensor(y, dtype= float32)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        sample = self.X[index]
        label = self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample, label

# Generate data per fold on synthetic dataset   
def GetDataFold(data_input, label_encoded_input, n_fold, fold_id, gen_size):
    test_start_idx = fold_id * gen_size
    test_end_idx = (fold_id + 1) * gen_size
    val_start_idx = (fold_id + 1) % n_fold * gen_size
    val_end_idx = (fold_id + 2) % n_fold * gen_size
    if val_end_idx == 0 :
        val_end_idx = len(data_input)

    X_test = data_input[test_start_idx:test_end_idx]
    y_test = label_encoded_input[test_start_idx:test_end_idx]

    X_val = data_input[val_start_idx:val_end_idx]
    y_val = label_encoded_input[val_start_idx:val_end_idx]

    if fold_id != n_fold - 1:
        X_train = np.concatenate((data_input[:test_start_idx], data_input[test_end_idx:val_start_idx], data_input[val_end_idx:]))
        y_train = np.concatenate((label_encoded_input[:test_start_idx], label_encoded_input[test_end_idx:val_start_idx], 
                                  label_encoded_input[val_end_idx:]))
    else:
        X_train = np.concatenate((data_input[:val_start_idx], data_input[val_end_idx:test_start_idx], data_input[test_end_idx:]))
        y_train = np.concatenate((label_encoded_input[:val_start_idx], label_encoded_input[val_end_idx:test_start_idx], 
                                  label_encoded_input[test_end_idx:]))
        
    return  X_train, y_train, X_val, y_val, X_test, y_test


# Generate data for test in specific fold
def GetDataTest(data_input, gt_input, label_encoded, fold_id, gen_size):
    test_start_idx = fold_id * gen_size
    test_end_idx = (fold_id + 1) * gen_size

    X_test = data_input[test_start_idx:test_end_idx]
    y_test = label_encoded[test_start_idx:test_end_idx]
    gt_test = gt_input[test_start_idx:test_end_idx]
        
    return  X_test, gt_test, y_test


# Get ready to use input data for training
def GetSubjectTrainData (data_dict, label_dict, subject_fold_id):
    train_id = subject_fold_id['train']
    val_id = subject_fold_id['val']
    test_id = subject_fold_id['test']

    X_train, y_train = GetSubjectData (data_dict, label_dict, train_id)
    X_val, y_val = GetSubjectData (data_dict, label_dict, val_id)
    X_test, y_test = GetSubjectData (data_dict, label_dict, test_id)
    
    return X_train,y_train,X_val,y_val,X_test,y_test

# Generate windowed data based on its id
def GetSubjectData (data_dict, label_dict, subject_id):
    X_wind, y_wind = [], []

    for test_sbj in subject_id:
        for window in (data_dict[test_sbj]):
            X_wind.append(window)
        for lbl in (label_dict[test_sbj]):
            y_wind.append(lbl)

    X_wind = np.array(X_wind) 
    y_wind = np.array(y_wind)
    
    return X_wind, y_wind

# Data Loader for training
def GetDataLoader(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
    data_train = DataGenerator(X_train, y_train)
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)

    data_val = DataGenerator(X_val, y_val)
    val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=True)

    data_test = DataGenerator(X_test, y_test)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False)
   
    print(f'Train dataset size: {X_train.shape}')
    print(f'Validation dataset size: {X_val.shape}')
    print(f'Test dataset size: {X_test.shape}')
    print(f'Batch size: {batch_size}')

    return train_loader, val_loader, test_loader


#Adjust dimension 
def AdjustDimenson(X_train, X_test, X_val):
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    X_val = np.expand_dims(X_val, axis=1)

    return X_train, X_test, X_val


#-----------------------------------------------------------------------------#
# TRAINING AND EVALUATION

#Early stopping training
class EarlyStoppingTrain:
    def __init__(self, patience=5, path='checkpoint.pth', trace_func=print):
        self.patience = patience
        self.counter = 0
        self.best_score = float('inf')
        self.early_stop = False
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, best_val, model):
        if val_loss >= best_val :
            self.counter += 1
            #self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
            self.trace_func(f'validation loss decreased ({best_val:.6f} --> {val_loss:.6f}).  Saving best model ...')
            save(model.state_dict(), self.path)

# Select optimizer
def GetOptimizer(optimizer_name, model_parameters, learning_rate, weight_decay=1e-3): #adjust weight if needed
    if optimizer_name == "Adam":
        return optim.Adam(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        return optim.SGD(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "RMSprop":
        return optim.RMSprop(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "Adagrad":
        return optim.Adagrad(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "Adamax":
        return optim.Adagrad(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

 # Training process           
def TrainingModel (train_model, train_loader, val_loader, num_epochs, patience, opt, lr, temp_path, model_path):
    my_device = device('cuda' if cuda.is_available() else 'cpu')
    train_model.to(my_device)
    early_stopping = EarlyStoppingTrain(patience=patience, path=temp_path)

    optimizer = GetOptimizer(opt, train_model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    loss_val_epoch = float('inf')
    loss_train_epoch = float('inf')
    acc_val_loss = -float('inf')
    acc_train_loss = -float('inf')

    # Training loop
    for epoch in range(num_epochs):
        train_model.train()
        running_loss = 0.0
        all_preds_train = []
        all_labels_train = []
    
        for batch_idx, (data, targets) in enumerate(train_loader):
            print(f"\rEpoch-{epoch+1}: {batch_idx+1}/{len(train_loader)} batch", end="")
            data = data.to(my_device)
            targets = targets.to(my_device)

            optimizer.zero_grad()
            output = train_model(data)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = max(output, 1)
            all_preds_train.extend(preds.cpu().numpy())
            labels = argmax(targets, dim=1)
            all_labels_train.extend(labels.cpu().numpy())
        
        # Calculate accuracy for this epoch
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = accuracy_score(all_labels_train, all_preds_train)
        print(f' -- training loss: {epoch_train_loss:.6f}, training accuracy: {epoch_train_acc:.6f}', end="")

        # Validation
        train_model.eval()
        val_loss = 0.0
        all_preds_val = []
        all_labels_val = []

        with no_grad():
            for data, targets in val_loader:
                data = data.to(my_device)
                targets = targets.to(my_device)
                outputs = train_model(data)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                _, preds = max(outputs, 1)
                all_preds_val.extend(preds.cpu().numpy())
                labels = argmax(targets, dim=1)
                all_labels_val.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = accuracy_score(all_labels_val, all_preds_val)
        print(f' -- eval loss: {epoch_val_loss:.6f}, accuracy: {epoch_val_acc:.6f}')
                      
        # Save the best model 
        early_stopping(epoch_val_loss, best_val_loss, train_model) #save best_model overall 
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss

        if epoch_val_loss < loss_val_epoch:
            loss_train_epoch = epoch_train_loss
            loss_val_epoch = epoch_val_loss
            acc_train_loss = epoch_train_acc
            acc_val_loss = epoch_val_acc
            save(train_model.state_dict(), model_path) #save best fold model
            print(f'Saving fold best model at epoch {epoch}')

        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    return loss_train_epoch, acc_train_loss, loss_val_epoch, acc_val_loss


# Model evaluation
def EvalModel (model, data_loader, disp=True):
  my_device = device('cuda:0' if cuda.is_available() else 'cpu')
  model.to(my_device)
  model.eval()
  all_preds = []
  all_labels = []
  eval_loss=0
  criterion = nn.CrossEntropyLoss()

  with no_grad():
    for data, targets in data_loader:
        data = data.to(my_device)
        targets = targets.to(my_device)
        outputs = model(data)
        loss = criterion(outputs, targets)
        eval_loss += loss.item()
       
        _, preds = max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        labels = argmax(targets, dim=1)
        all_labels.extend(labels.cpu().numpy())

  eval_loss = eval_loss / len(data_loader)
  acc = accuracy_score(all_labels, all_preds)
  f1 = f1_score(all_labels, all_preds, average='weighted')
  precision = precision_score(all_labels, all_preds, average='weighted')
  recall = recall_score(all_labels, all_preds, average='weighted')
  eval_result = [eval_loss, acc, f1, precision, recall]

  if disp :
    print(f' -- eval loss: {eval_loss:.6f}, accuracy: {acc:.6f}')
    print(f' -- f1-score: {f1:.6f}, precision: {precision:.6f}, recall: {recall:.6f}')

  return eval_result, all_preds


#-----------------------------------------------------------------------------#
# VISUALISATION AND ANALYSIS

# Create heatmap
def HeatmapCM(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    return fig

# Extract weights from a model
def GetModelWeights(model):
    weights_dict = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights_dict[name] = param.detach().cpu().numpy()
    return weights_dict

# Visualize weights as heatmaps
def VisualizeWeights(before_weights, after_weights):
    layers = list(before_weights.keys())
    num_layers = len(layers)

    fig, axs = plt.subplots(num_layers, 2, figsize=(10, num_layers * 3))
    fig.suptitle('Weights Before and After Randomization', fontsize=16)

    for i, layer in enumerate(layers):
        # Average over the batch dimension
        before_weights_avg = before_weights[layer].squeeze()
        after_weights_avg = after_weights[layer].squeeze()

        if len(before_weights_avg.shape) > 2:
            before_weights_avg = before_weights_avg.mean(axis=0)
            after_weights_avg = after_weights_avg.mean(axis=0)

        heatmap(before_weights_avg, ax=axs[i, 0], cmap='coolwarm', cbar=True)
        axs[i, 0].set_title(f'{layer} (Before)')
        axs[i, 0].set_xlabel('Output Channels')
        axs[i, 0].set_ylabel('Input Channels')
        heatmap(after_weights_avg, ax=axs[i, 1], cmap='coolwarm', cbar=True)
        axs[i, 1].set_title(f'{layer} (After)')
        axs[i, 1].set_xlabel('Output Channels')
        axs[i, 1].set_ylabel('Input Channels')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig

# Randomize weights for robustness evaluation
def RandomiseWeights(model, num_layer= None):
    print("Randomising weights for model...")
    rand_model = deepcopy(model)
    conv_layers = [layer for layer in rand_model.children() if isinstance(layer, nn.Conv2d)]
    
    if num_layer == None :
        for layer in conv_layers[:num_layer]:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
                print(f"Randomized weights for layer: {layer}")
    else :
        sample_idx = (np.arange(len(conv_layers)), num_layer)
        for idx in sample_idx:
            layer = conv_layers[idx]
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
                print(f"Randomized weights for layer: {layer}")

    return rand_model
