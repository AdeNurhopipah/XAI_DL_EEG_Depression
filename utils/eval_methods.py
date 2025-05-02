"""
This module provides metrics for evaluating the similarity and robustness of attribution maps in EEG-based 
explainable AI.

Functions:
    - calculate_rma: Computes Relevance Mass Accuracy (RMA) to measure overlap between attribution and ground truth.
    - calculate_cosine_similarity: Computes cosine similarity between attribution and ground truth maps.
    - calculate_pearson_cor: Computes Pearson correlation to assess robustness between original and perturbed maps.
    - calculate_ssim: Computes Structural Similarity Index (SSIM) between original and perturbed maps.

Author: Ade Nurhopipah
Email: ade.nurhopipah@postgrad.otago.ac.nz
"""

from numpy import sum, linalg, dot, ravel, std, corrcoef, max, min, isnan, mean
from sklearn.metrics.pairwise import cosine_similarity
from torch import tensor, float32, isnan


# Relevance Mass Accuracy (RMA)
def GetRMA(attribution, gt_data):
    epsilon = 1e-9
    within = sum((gt_data == 1) & (attribution == 1))
    total = sum(attribution == 1)
    if total == 0:
        print("Warning: No positive values in attribution map.")
        return 0  # Default to 0 if there are no relevant attributions
    mass_similarity = within / total + epsilon
    return mass_similarity

# Cosine Similarity
def GetCosineSim(attribution, gt_data):

    # Flatten the matrices into vectors
    vector1 = attribution.flatten().reshape(1, -1)
    vector2 = gt_data.flatten().reshape(1, -1)
    
    # Use sklearn to calculate cosine similarity
    similarity = cosine_similarity(vector1, vector2)[0, 0]
    
    return similarity


# Pearson corellation for robustness test
def GetPearsonCor(ori_att, rand_att):
    # Ensure inputs are 1D arrays
    ori_att = ravel(ori_att)
    rand_att = ravel(rand_att)

    # Check for zero variance
    if std(ori_att) == 0 or std(rand_att) == 0:
        print("Warning: Zero variance in input arrays. Returning PC 0.")
        return 0.0

    try:
        # Compute Pearson correlation
        correlation_matrix = corrcoef(ori_att, rand_att)
        correlation = correlation_matrix[0, 1]
        if isnan(tensor(correlation)):
            print("Warning: SSIM result is NaN. Returning SSIM as 0.")
            return 0.0
    except Exception as e:
        print(f"Error in Pearson correlation calculation: {e}")
        return 0.0

    return correlation

# Structural Similarity Index Measure (SSIM) for robustness test
def GetSSIM(ori_att, rand_att):
    from pytorch_msssim import ssim
    # Convert inputs to tensors
    ori_att = tensor(ori_att, dtype=float32).unsqueeze(0).unsqueeze(0)
    rand_att = tensor(rand_att, dtype=float32).unsqueeze(0).unsqueeze(0)

    data_range=rand_att.max() - rand_att.min()
    if data_range == 0:
        print("Warning: No variation in the original attribution map. Returning SSIM as 0.")
        return 0.0
    
    try:
        ssim_value = ssim(ori_att, rand_att, data_range=data_range).item()
        if isnan(tensor(ssim_value)):
            print("Warning: SSIM result is NaN. Returning SSIM as 0.")
            return 0.0
    except Exception as e:
        print(f"Error in SSIM calculation with PyTorch: {e}")
        return 0.0
    
    return ssim_value


# Select xai methods
def EvalScore (imp_score, comp_score, method):
    if method == 'RMA' :
        eval_score = GetRMA(imp_score, comp_score)
    elif method == 'CS' :
        eval_score = GetCosineSim(imp_score, comp_score)
    elif method == 'PC' :
        eval_score = GetPearsonCor(imp_score, comp_score)
    elif method == 'SSIM' :
        eval_score = GetSSIM(imp_score, comp_score)
    else :
        print('Method not found')

    return eval_score
